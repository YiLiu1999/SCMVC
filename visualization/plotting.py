import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE


CLASSIFICATION_COLORS = [
    "black", "yellow", "lightgreen", "indigo", "orange", "pink", "peru",
    "crimson", "aqua", "dodgerblue", "slategrey", "b", "red", "darkcyan",
    "grey", "olive", "green", "gold",
]

TSNE_COLORS = CLASSIFICATION_COLORS[1:]


def draw_classification_map(pred, label, name, acc, save_dir='./results',
                            scale=4.0, dpi=400):
    """Draw and save the classification map.

    Args:
        pred: Predicted labels (1-indexed, 0 is background)
        label: Ground truth map (2D array, 0 is background)
        name: Dataset name for save path
        acc: Accuracy value for filename
        save_dir: Directory to save the result
        scale: Figure scale factor
        dpi: Figure DPI
    """
    indices = np.where(label != 0)
    label[indices] = pred

    rgb_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(CLASSIFICATION_COLORS):
        rgb = np.array(mcolors.to_rgb(color)) * 255
        rgb_label[label == i] = rgb.astype(np.uint8)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(rgb_label)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.savefig(
        f'{save_dir}/{name}/pred_{acc:.5f}.png',
        format='png', transparent=True, dpi=dpi, pad_inches=0
    )


def draw_tsne(features, labels, acc, dataname, save_dir='./results', title=None):
    """Draw and save the t-SNE visualization.

    Args:
        features: Feature tensor from the model
        labels: Ground truth labels
        acc: Accuracy value for filename
        dataname: Dataset name for save path
        save_dir: Directory to save the result
        title: Optional plot title
    """
    X = features.cpu().numpy()
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    for label_val in unique_labels:
        mask = labels == label_val
        color_idx = int(label_val) % len(TSNE_COLORS)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], marker='o',
                    color=TSNE_COLORS[color_idx], s=10)

    if title is not None:
        plt.title(title)

    plt.rc('font', family='Times New Roman')
    plt.savefig(
        f'{save_dir}/{dataname}/tsne_{acc:.5f}.pdf',
        bbox_inches='tight', pad_inches=0
    )
    plt.show()
