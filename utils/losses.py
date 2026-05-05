import torch


def contrastive_loss(z_i, z_j, temperature):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.

    Args:
        z_i: Projected features from view i, shape [B, D]
        z_j: Projected features from view j, shape [B, D]
        temperature: Temperature scaling parameter

    Returns:
        Scalar contrastive loss
    """
    batch_size = z_i.shape[0]
    out = torch.cat([z_i, z_j], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss
