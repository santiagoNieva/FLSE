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
    device: str | None = None,
    log_every: int | None = None,
):
    """
    Entrena el modelo FLSE por distillation sobre todos los índices de vocabulario.
    Ajusta automáticamente el dispositivo si no se especifica.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    teacher_vectors = teacher_vectors.to(device)
    target_entropies = target_entropies.to(device)

    vocab_size = teacher_vectors.size(0)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    history = {"loss": [], "mse": [], "ent": []}

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

            if log_every and (i // batch_size) % log_every == 0:
                print(
                    f"[epoch {epoch+1} step {i}] "
                    f"loss={loss.item():.4f} mse={mse.item():.4f} ent={ent_loss.item():.4f}"
                )

        avg_loss = total_loss / vocab_size
        avg_mse = total_mse / vocab_size
        avg_ent = total_ent / vocab_size

        history["loss"].append(avg_loss)
        history["mse"].append(avg_mse)
        history["ent"].append(avg_ent)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- Total: {avg_loss:.4f} "
            f" MSE: {avg_mse:.4f} "
            f" EntReg: {avg_ent:.4f}"
        )

    return model, history
