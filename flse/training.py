import torch
from torch import optim
from .losses import mse_loss_with_entropy

def train_distillation(
    model,
    teacher_vectors: torch.Tensor,
    target_entropies: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-2,
    lambda_ent: float = 0.1,
    device: str = "cuda",
):
    vocab_size = teacher_vectors.size(0)
    indices = torch.arange(vocab_size, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = torch.randperm(vocab_size, device=device)
        total_loss = total_mse = total_ent = 0.0

        for i in range(0, vocab_size, batch_size):
            batch_idx = perm[i:i+batch_size]

            optimizer.zero_grad()
            loss, mse, ent_loss = mse_loss_with_entropy(
                model,
                batch_idx,
                teacher_vectors,
                target_entropies,
                lambda_ent=lambda_ent,
            )
            loss.backward()
            optimizer.step()

            bs = batch_idx.size(0)
            total_loss += loss.item() * bs
            total_mse  += mse.item() * bs
            total_ent  += ent_loss.item() * bs

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- Total: {total_loss / vocab_size:.4f} "
            f" MSE: {total_mse / vocab_size:.4f} "
            f" EntReg: {total_ent / vocab_size:.4f}"
        )
