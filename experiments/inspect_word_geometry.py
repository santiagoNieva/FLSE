# experiments/inspect_word_geometry.py

import torch
from flse.model import FLSEModel
from flse.geometry import generate_vertices


def inspect_word(model, word_idx, topk=5):
    logits = model.logits[word_idx]  # (L, V)
    weights = torch.softmax(logits, dim=-1)

    print(f"\nPalabra idx={word_idx}")
    for l in range(weights.shape[0]):
        w = weights[l]
        top_vals, top_idx = torch.topk(w, topk)
        print(f"  Capa {l}:")
        for v, i in zip(top_vals.tolist(), top_idx.tolist()):
            print(f"    vértice {i:2d}  peso={v:.3f}")


if __name__ == "__main__":
    # modelo toy
    vocab_size = 100
    vertices = generate_vertices(3, 16, 8)
    model = FLSEModel(vocab_size, vertices, teacher_dim=32)

    # sin entrenar (random)
    inspect_word(model, word_idx=10)
