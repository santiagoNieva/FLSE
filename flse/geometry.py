import numpy as np
import torch

def _normalize_np(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def generate_vertices(num_layers: int, verts_per_layer: int, dim: int) -> torch.Tensor:
    """
    Genera vértices esféricos para todas las capas.
    Devuelve tensor (L, V, D) en torch.float32.
    """
    layers = []
    for _ in range(num_layers):
        vs = np.random.normal(size=(verts_per_layer, dim))
        vs = _normalize_np(vs)
        layers.append(vs)
    vertices = np.stack(layers)  # (L, V, D)
    return torch.tensor(vertices, dtype=torch.float32)
