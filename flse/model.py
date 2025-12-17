import torch
import torch.nn as nn
import torch.nn.functional as F

class FLSEModel(nn.Module):
    """
    Modelo base FLSE: solo sabe
    - combinar vértices por capa usando logits
    - proyectar al espacio del teacher (si se define una cabeza)
    """
    def __init__(
        self,
        vocab_size: int,
        vertices: torch.Tensor,
        teacher_dim: int,
        logit_temps: torch.Tensor | None = None,
    ):
        super().__init__()
        # vertices: (L, V, D)
        self.vertices = nn.Parameter(vertices, requires_grad=False)
        self.num_layers, self.verts_per_layer, self.dim = vertices.shape

        # temperaturas por capa para escalar logits antes del softmax (1.0 = sin cambio)
        if logit_temps is None:
            logit_temps = torch.ones(self.num_layers, dtype=vertices.dtype)
        elif logit_temps.numel() != self.num_layers:
            raise ValueError("logit_temps debe tener num_layers valores.")
        self.register_buffer("logit_temps", logit_temps, persistent=False)

        # logits por palabra/capa/vértice: (Vocab, L, V)
        self.logits = nn.Parameter(
            torch.zeros(vocab_size, self.num_layers, self.verts_per_layer)
        )

        # proyección opcional al espacio del teacher
        self.head = nn.Linear(self.num_layers * self.dim, teacher_dim)

    def forward(self, idx_batch: torch.Tensor) -> torch.Tensor:
        """
        idx_batch: (B,) índices de vocab.
        Devuelve: (B, teacher_dim)
        """
        logits = self.logits[idx_batch]              # (B, L, V)
        temps = self.logit_temps.view(1, self.num_layers, 1)
        weights = F.softmax(logits * temps, dim=-1)  # (B, L, V)

        verts = self.vertices.unsqueeze(0)           # (1, L, V, D)
        weights_exp = weights.unsqueeze(-1)          # (B, L, V, 1)

        rep_per_layer = torch.sum(weights_exp * verts, dim=2)   # (B, L, D)
        rep = rep_per_layer.reshape(rep_per_layer.size(0), -1)  # (B, L*D)

        out = self.head(rep)                         # (B, teacher_dim)
        return out

    def flse_embedding(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Devuelve el embedding fractal (sin la cabeza lineal).
        idx: escalar o (B,)
        """
        if idx.dim() == 0:
            logits = self.logits[idx]               # (L, V)
            temps = self.logit_temps
            weights = F.softmax(logits * temps, dim=-1)     # (L, V)
            rep_per_layer = torch.sum(
                weights.unsqueeze(-1) * self.vertices, dim=1
            )                                       # (L, D)
            return rep_per_layer.reshape(-1)        # (L*D,)
        else:
            logits = self.logits[idx]               # (B, L, V)
            temps = self.logit_temps.view(1, self.num_layers, 1)
            weights = F.softmax(logits * temps, dim=-1)     # (B, L, V)
            verts = self.vertices.unsqueeze(0)      # (1, L, V, D)
            rep_per_layer = torch.sum(
                weights.unsqueeze(-1) * verts, dim=2
            )                                       # (B, L, D)
            return rep_per_layer.reshape(rep_per_layer.size(0), -1)
