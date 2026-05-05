import torch


class KMeans:
    """K-Means clustering implemented in PyTorch for anchor center initialization."""

    def __init__(self, n_clusters=10, max_iter=None, verbose=True, device=torch.device("cpu")):
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        """Fit K-Means on input features and return cluster assignments."""
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        self.centers = x[init_row]
        while True:
            self.nearest_center(x)
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, 0))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break
            self.count += 1
        return self.get_assignments()

    def nearest_center(self, x):
        """Assign each sample to its nearest cluster center."""
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum((sample - self.centers) ** 2, dim=1)
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], dim=0)
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        """Update cluster centers as the mean of assigned samples."""
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, dim=0).unsqueeze(0)], dim=0)
        self.centers = centers

    def get_assignments(self):
        """Return cluster assignment for each sample."""
        return torch.argmin(self.dists, dim=1)
