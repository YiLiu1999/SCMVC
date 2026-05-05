import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .spatial_encoder import ResNet50
from .spectral_encoder import SpectralTransformer


class SCMVC(nn.Module):
    """
    Semantic Constraint-based Spatial-Spectral Multiview Clustering (SCMVC).

    Combines a ResNet-50 spatial encoder and a Transformer spectral encoder
    for multiview feature fusion and clustering of hyperspectral images.
    """

    def __init__(self, feature_dim, n_spectral_bands, n_anchors, embedding_dim):
        super().__init__()
        self.spatial_encoder = ResNet50(feature_dim)
        self.spectral_encoder = SpectralTransformer(
            n_spectral_bands, 4 * feature_dim, feature_dim, 0, 0, 6
        )
        self.anchor_centers = Parameter(
            torch.Tensor(n_anchors, embedding_dim), requires_grad=True
        )

    def forward(self, spatial_input, spectral_input):
        spatial_feat, spatial_proj = self.spatial_encoder(spatial_input)
        spectral_feat, spectral_proj = self.spectral_encoder(spectral_input)

        fused_feat = (spatial_feat + spectral_feat) * 1e2 / 2
        fused_proj = (spatial_proj + spectral_proj) / 2

        similarity_matrix = F.softmax(
            torch.mm(fused_proj, fused_proj.t().contiguous()) / fused_proj.shape[0],
            dim=1
        )
        return fused_feat, fused_proj, similarity_matrix
