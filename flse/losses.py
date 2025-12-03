import torch
import torch.nn.functional as F
import math

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)

def batch_entropy_per_layer(model, batch_idx: torch.Tensor) -> torch.Tensor:
    """
    Entropía promedio por capa para un batch.
    Devuelve (L,)
    """
    logits = model.logits[batch_idx]               # (B, L, V)
    weights = F.softmax(logits, dim=-1)            # (B, L, V)
    ent = -torch.sum(weights * torch.log(weights + 1e-9), dim=-1)  # (B, L)
    return ent.mean(dim=0)

def mse_loss_with_entropy(
    model,
    idx_batch: torch.Tensor,
    teacher_vectors: torch.Tensor,
    target_entropies: torch.Tensor,
    lambda_ent: float = 0.1,
):
    """
    Loss combinada:
    - MSE entre salida FLSE y teacher
    - Penalización de entropía por capa
    """
    pred = model(idx_batch)
    target = teacher_vectors[idx_batch]
    mse = mse_loss(pred, target)

    ent = batch_entropy_per_layer(model, idx_batch)
    ent_loss = torch.mean((ent - target_entropies)**2)

    total = mse + lambda_ent * ent_loss
    return total, mse.detach(), ent_loss.detach()
