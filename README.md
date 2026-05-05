<h1 align="center">SCMVC</h1>

<p align="center"><strong>Semantic Constraint-Based Spatial-Spectral Multiview Clustering for Hyperspectral Images</strong></p>

<p align="center">
  <b>Author A</b><sup>1</sup> &nbsp; <b>Author B</b><sup>1</sup> &nbsp; <b>Author C</b><sup>2</sup><code>*</code>
</p>


<p align="center">
  <sup>1</sup> Affiliation 1 &nbsp;&nbsp;|&nbsp;&nbsp; <sup>2</sup> Affiliation 2
</p>


<p align="center">
  <small><code>*</code> Corresponding Author</small>
</p>


<p align="center">
  <a href="https://doi.org/YOUR_DOI"><img src="https://img.shields.io/badge/Paper-DOI-0085ca?style=flat-square" alt="DOI"></a>
  &nbsp;
  <a href="https://github.com/YOUR_REPO"><img src="https://img.shields.io/badge/Code-SCMVC-green?style=flat-square" alt="Code"></a>
</p>


<p align="center">
  If you find this project helpful, please consider giving it a <strong>⭐ star</strong>!
</p>


---

## 📖 Introduction

Learning discriminative representations from **hyperspectral images (HSI)** is crucial for unsupervised clustering. Existing multi-view contrastive methods align view-level features but often neglect the **semantic consistency** of the learned similarity structures across views, leading to suboptimal clustering boundaries. To address this, we propose **SCMVC** (Semantic Constraint-based Spatial-Spectral Multiview Clustering):

- **Spatial-spectral multiview construction** — Spectral bands are split into two RGB sub-band groups for spatial views processed by **ResNet-50**; the full spectrum at each center pixel is processed by a **Transformer** to capture global spectral dependencies.
- **Contrastive alignment (NT-Xent)** — Pulls together projections from different views of the same sample while pushing apart those from different samples.
- **Semantic constraint loss** — Enforces cross-entropy consistency on the similarity matrices produced by the two spatial views, encouraging the network to preserve the same semantic neighbourhood structure regardless of the view.
- **Anchor-based clustering** — Learnable anchor centers are updated via **K-Means** during training; final labels are assigned by similarity to anchor centers **without post-processing**.

Experiments on **seven** benchmark HSI datasets (Indian Pines, Pavia University, Salinas, Botswana, Houston, Yancheng, HanChuan) demonstrate the effectiveness of SCMVC.

---

## 🏗️ Method Overview

![image-20260505215611317](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20260505215611317.png)

---

## 📂 Repository Structure

```
SCMVC/
├── main.py                                 # Training and evaluation entry point
├── README.md
├── models/
│   ├── __init__.py
│   ├── scmvc.py                            # SCMVC model (spatial-spectral fusion)
│   ├── spatial_encoder.py                  # ResNet-50 spatial encoder
│   └── spectral_encoder.py                 # Transformer spectral encoder
├── utils/
│   ├── __init__.py
│   ├── dataset.py                          # HSIDataset class and data augmentation
│   ├── metrics.py                          # Clustering metrics (ACC, NMI, ARI, etc.)
│   ├── losses.py                           # NT-Xent contrastive loss
│   └── kmeans.py                           # K-Means clustering (PyTorch)
└── visualization/
    ├── __init__.py
    ├── plotting.py                         # Classification map and t-SNE visualization
    └── parameter_analysis.py               # Hyperparameter sensitivity analysis
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_REPO/SCMVC.git
cd SCMVC

# Create environment (Python >= 3.8)
conda create -n scmvc python=3.10
conda activate scmvc

# Install dependencies
pip install torch torchvision
pip install h5py scipy numpy scikit-learn tqdm pandas matplotlib Pillow
```

---

## 💡 Data Preparation & Usage

Use the preprocessing script from [create_dataset.py](https://github.com/YiLiu1999/EMVCC/tree/main/dataset/create_dataset.py) to convert raw `.mat` files into HDF5 format.

Each `.h5` file should contain:

- `data`: shape `(N, 28*28*B)` — flattened 28×28 spatial-spectral patches (first 6 channels for two spatial views, remaining for spectral view)
- `label`: shape `(N,)` — per-sample class index (0-based)

Place all `.h5` and ground truth `.mat` files in the data directory (default: `./data/`).

```
data/
  ├── IP-28-28-206.h5
  ├── Indian_pines_gt.mat
  ├── pu-28-28-109.h5
  ├── PaviaU_gt.mat
  └── ...
```

### Training

```bash
# Train on Houston University (default)
python main.py --dataset houstonu --data_dir ./data --gt_dir ./data

# Train on Indian Pines
python main.py --dataset indian --data_dir ./data --gt_dir ./data

# Train on Salinas with custom hyperparameters
python main.py --dataset salinas --data_dir ./data --gt_dir ./data \
    --lr 0.00001 --alpha 1e-3 --epochs 300 --batch_size 256

# Specify GPU device
python main.py --dataset botswana --data_dir ./data --gt_dir ./data --gpu 0
```

### Output

Training outputs are saved to `./results/<dataset>/`:

```
results/<dataset>/
  ├── best_model.pth           # Best model weights
  ├── best_label.txt           # Predicted cluster labels
  ├── training_statistics.csv  # Loss and accuracy per epoch
  ├── training_loss.png        # Training loss curve
  ├── accuracy.png             # Clustering accuracy curve
  ├── pred_<acc>.png           # Classification map
  └── tsne_<acc>.pdf           # t-SNE visualization
```

---

## 📊 Datasets & Implementation Details

| Dataset          | Clusters | Samples | Views | Bands |
| ---------------- | -------- | ------- | ----- | ----- |
| Indian Pines     | 16       | 10,249  | 3     | 206   |
| Pavia University | 9        | 42,776  | 3     | 109   |
| Salinas          | 16       | 54,129  | 3     | 230   |
| Botswana         | 14       | 3,248   | 3     | 151   |
| Houston          | 15       | 15,029  | 3     | 150   |
| Yancheng         | 18       | 7,894   | 3     | 259   |
| HanChuan         | 16       | 257,530 | 3     | 280   |

**Implementation:** ResNet-50 (spatial) + Transformer with 6 attention heads (spectral); projection head dimension 128; Adam optimizer with weight_decay=1e-6; dataset-specific learning rate and α.

| Dataset          | Learning Rate | α    | Temperature |
| ---------------- | ------------- | ---- | ----------- |
| Indian Pines     | 1e-3          | 1e-3 | 0.5         |
| Pavia University | 1e-3          | 1e-3 | 0.5         |
| Salinas          | 1e-5          | 1e-3 | 0.5         |
| Botswana         | 1e-4          | 1e-3 | 0.5         |
| Houston          | 1e-4          | 1e-2 | 1.0         |
| Yancheng         | 1e-4          | 1e-2 | 1.0         |
| HanChuan         | 1e-3          | 1e-3 | 0.5         |

---

## 📜 Citation

If you use this code or the paper in your research, please cite:

```bibtex
@ARTICLE{11223688,
  author={Luo, Fulin and Liu, Yi and Guo, Tan and Fu, Chuan and Duan, Yule and Shi, Qian and Du, Bo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SCMVC: Semantic Constraint-Based Spatial–Spectral Multiview Clustering for Hyperspectral Images}, 
  year={2025},
  volume={63},
  pages={1-13},
}

```

