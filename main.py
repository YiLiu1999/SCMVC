import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import h5py
import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

from models import SCMVC
from utils import HSIDataset, train_transform, test_transform, evaluate, contrastive_loss, KMeans
from visualization import draw_classification_map, draw_tsne

warnings.filterwarnings('ignore')

# Dataset configurations: n_classes, n_samples, n_bands, lr, alpha, temperature, h5_file, gt_file, gt_key
DATASET_CONFIG = {
    'indian': {
        'n_classes': 16, 'n_samples': 10249, 'n_bands': 206,
        'lr': 0.001, 'alpha': 1e-3, 'temperature': 0.5,
        'h5_file': 'IP-28-28-206.h5',
        'gt_file': 'Indian_pines_gt.mat', 'gt_key': 'indian_pines_gt',
    },
    'paviau': {
        'n_classes': 9, 'n_samples': 42776, 'n_bands': 109,
        'lr': 0.001, 'alpha': 1e-3, 'temperature': 0.5,
        'h5_file': 'pu-28-28-109.h5',
        'gt_file': 'PaviaU_gt.mat', 'gt_key': 'paviaU_gt',
    },
    'salinas': {
        'n_classes': 16, 'n_samples': 54129, 'n_bands': 230,
        'lr': 0.00001, 'alpha': 1e-3, 'temperature': 0.5,
        'h5_file': 'Sa-28-28-230.h5',
        'gt_file': 'Salinas_gt.mat', 'gt_key': 'salinas_gt',
    },
    'botswana': {
        'n_classes': 14, 'n_samples': 3248, 'n_bands': 151,
        'lr': 0.0001, 'alpha': 1e-3, 'temperature': 0.5,
        'h5_file': 'Bw-28-28-151.h5',
        'gt_file': 'Botswana_gt.mat', 'gt_key': 'Botswana_gt',
    },
    'houstonu': {
        'n_classes': 15, 'n_samples': 15029, 'n_bands': 150,
        'lr': 0.0001, 'alpha': 1e-2, 'temperature': 1.0,
        'h5_file': 'HU-28-28-200.h5',
        'gt_file': 'HoustonU.mat', 'gt_key': 'HoustonU_GT',
    },
    'yancheng': {
        'n_classes': 18, 'n_samples': 7894, 'n_bands': 259,
        'lr': 0.0001, 'alpha': 1e-2, 'temperature': 1.0,
        'h5_file': 'yc-28-28-259.h5',
        'gt_file': None, 'gt_key': None,
    },
    'hanchuan': {
        'n_classes': 16, 'n_samples': 257530, 'n_bands': 280,
        'lr': 0.001, 'alpha': 1e-3, 'temperature': 0.5,
        'h5_file': 'HC-28-28-280.h5',
        'gt_file': 'WHU_Hi_HanChuan_gt.mat', 'gt_key': 'WHU_Hi_HanChuan_gt',
    },
}


def setup_seed(seed):
    """Fix random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def extract_features(model, data_loader, device):
    """Extract fused spatial-spectral features from the entire dataset."""
    model.eval()
    feature_bank = []
    with torch.no_grad():
        for spatial_v1, spatial_v2, spectral_v, _ in tqdm(data_loader, desc='Extracting features'):
            spatial_v1 = spatial_v1.to(device)
            spatial_v2 = spatial_v2.to(device)
            spectral_v = spectral_v.to(device)
            feat1, _, _ = model(spatial_v1, spectral_v)
            feat2, _, _ = model(spatial_v2, spectral_v)
            feature_bank.append(feat1 + feat2)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
    return feature_bank


def train_epoch(model, data_loader, optimizer, temperature, alpha, device):
    """Train the SCMVC model for one epoch."""
    model.train()
    total_loss, total_num = 0.0, 0
    ce_criterion = nn.CrossEntropyLoss(reduction="sum")
    train_bar = tqdm(data_loader)

    for spatial_v1, spatial_v2, spectral_v, _ in train_bar:
        spatial_v1 = spatial_v1.to(device)
        spatial_v2 = spatial_v2.to(device)
        spectral_v = spectral_v.to(device)

        feat1, proj1, sim1 = model(spatial_v1, spectral_v)
        feat2, proj2, sim2 = model(spatial_v2, spectral_v)

        # Contrastive loss (NT-Xent)
        loss_contrastive = contrastive_loss(proj1, proj2, temperature)
        # Semantic constraint loss
        loss_semantic = ce_criterion(sim1, sim2) + ce_criterion(sim1.t(), sim2.t())

        loss = loss_contrastive + alpha * loss_semantic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = spatial_v1.size(0)
        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(f'Train Loss: {total_loss / total_num:.4f}')

    return total_loss / total_num


def evaluate_epoch(model, data_loader, n_classes, device):
    """Evaluate clustering: extract features, run K-Means, compute metrics."""
    model.eval()
    feature_bank = []
    with torch.no_grad():
        for spatial_v1, spatial_v2, spectral_v, _ in tqdm(data_loader, desc='Evaluating'):
            spatial_v1 = spatial_v1.to(device)
            spatial_v2 = spatial_v2.to(device)
            spectral_v = spectral_v.to(device)
            feat1, _, _ = model(spatial_v1, spectral_v)
            feat2, _, _ = model(spatial_v2, spectral_v)
            feature_bank.append(feat1 + feat2)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()

        target = torch.tensor(data_loader.dataset.label, device=device)

        # Anchor-based prediction
        sim_matrix = F.normalize(torch.mm(feature_bank, model.anchor_centers.t()), dim=-1)
        anchor_pred = sim_matrix.softmax(dim=1).argmax(dim=1)

        # K-Means clustering and anchor center update
        kmeans = KMeans(n_classes, max_iter=20, verbose=False, device=device)
        cluster_ids = kmeans.fit(feature_bank)
        model.anchor_centers.data = kmeans.centers

        # Evaluate K-Means clustering accuracy
        acc, label_mapping, ca = evaluate(target, cluster_ids)

        # Map anchor predictions to true label space
        pred_labels = np.array([label_mapping[int(l)] for l in anchor_pred.cpu().numpy()])

    return acc, pred_labels, feature_bank, ca


def main():
    parser = argparse.ArgumentParser(description='SCMVC: Semantic Constraint-based Multiview Clustering')
    parser.add_argument('--dataset', type=str, default='houstonu',
                        choices=list(DATASET_CONFIG.keys()), help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--gt_dir', type=str, default='./data', help='Path to ground truth directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dimension')
    parser.add_argument('--temperature', default=None, type=float, help='Temperature for contrastive loss')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
    parser.add_argument('--embedding', default=128, type=int, help='Embedding dimension for anchor centers')
    parser.add_argument('--alpha', default=None, type=float, help='Weight for semantic constraint loss')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device ID')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--seed', default=3307, type=int, help='Random seed')
    args = parser.parse_args()

    # Device setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)

    # Load dataset configuration
    config = DATASET_CONFIG[args.dataset]
    n_classes = config['n_classes']
    n_samples = config['n_samples']
    n_bands = config['n_bands']
    lr = args.lr if args.lr is not None else config['lr']
    alpha = args.alpha if args.alpha is not None else config['alpha']
    temperature = args.temperature if args.temperature is not None else config['temperature']

    # Load HDF5 data
    h5_path = os.path.join(args.data_dir, config['h5_file'])
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]

    # Load ground truth map for visualization
    if args.dataset == 'yancheng':
        y1 = sio.loadmat(os.path.join(args.gt_dir, 'train_label.mat'))
        y2 = sio.loadmat(os.path.join(args.gt_dir, 'test_label.mat'))
        ground_truth_map = y1['train_label'] + y2['test_label']
    else:
        gt_path = os.path.join(args.gt_dir, config['gt_file'])
        ground_truth_map = sio.loadmat(gt_path)[config['gt_key']]

    # Create data loaders
    train_dataset = HSIDataset(data, label, n_bands, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    eval_dataset = HSIDataset(data, label, n_bands, transform=test_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Initialize model
    n_spectral_bands = n_bands - 6
    model = SCMVC(args.feature_dim, n_spectral_bands, n_classes, args.embedding).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # Initialize anchor centers via K-Means on pre-extracted features
    print('Initializing anchor centers...')
    feature_bank = extract_features(model, train_loader, device)
    kmeans = KMeans(n_classes, max_iter=20, verbose=False, device=device)
    kmeans.fit(feature_bank)
    model.anchor_centers.data = kmeans.centers
    print('Anchor centers initialized.')

    # Create results directory
    save_dir = f'./results/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    best_acc = 0
    best_label = None
    best_feature = None
    best_ca = None
    best_epoch = 0
    train_losses = []
    accuracies = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, temperature, alpha, device)
        acc, pred, features, ca = evaluate_epoch(model, eval_loader, n_classes, device)

        train_losses.append(train_loss)
        accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_label = pred
            best_feature = features
            best_ca = ca
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    # Print results
    print(f'Best accuracy: {best_acc:.4f} at epoch {best_epoch}')
    for i, ca_val in enumerate(best_ca):
        print(f'Class #{i} ACC: {ca_val:.4f}')

    # Save best labels
    if best_label is not None:
        label_str = '\n'.join(
            map(str, best_label.tolist() if isinstance(best_label, torch.Tensor) else best_label.tolist())
        )
        with open(os.path.join(save_dir, 'best_label.txt'), 'w') as f:
            f.write(label_str)

    # Save training statistics
    results_df = pd.DataFrame({'train_loss': train_losses, 'test_acc': accuracies})
    results_df.to_csv(os.path.join(save_dir, 'training_statistics.csv'), index_label='epoch')

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(accuracies)
    plt.title('Clustering Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))

    # Visualization
    draw_classification_map(best_label + 1, ground_truth_map,
                            args.dataset, best_acc, save_dir='./results')
    draw_tsne(best_feature, train_dataset.label, best_acc,
              args.dataset, save_dir='./results')


if __name__ == '__main__':
    main()
