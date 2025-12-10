# scripts/distill_from_fake_teacher.py

import argparse
import torch
import torch.nn.functional as F

from flse.geometry import generate_vertices
from flse.model import FLSEModel
from flse.training import train_distillation


def build_fake_teacher(vocab_size: int, teacher_dim: int, seed: int = 0):
    """
    Crea embeddings teacher sintéticos con estructura suave:
    clusters + ruido pequeño.
    """
    torch.manual_seed(seed)

    # Centros semánticos ficticios
    num_clusters = max(2, vocab_size // 10)
    cluster_centers = torch.randn(num_clusters, teacher_dim)
    cluster_centers = F.normalize(cluster_centers, dim=-1)

    assignments = torch.randint(0, num_clusters, (vocab_size,))
    noise = 0.05 * torch.randn(vocab_size, teacher_dim)

    teacher = cluster_centers[assignments] + noise
    teacher = F.normalize(teacher, dim=-1)

    return teacher


def main():
    parser = argparse.ArgumentParser("FLSE fake-teacher distillation")

    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--verts-per-layer", type=int, default=16)
    parser.add_argument("--layer-dim", type=int, default=8)
    parser.add_argument("--teacher-dim", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-ent", type=float, default=0.1)

    parser.add_argument(
        "--target-entropies",
        type=float,
        nargs="+",
        default=None,
        help="Lista de entropías objetivo por capa",
    )

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # --- teacher sintético ---
    teacher_vectors = build_fake_teacher(
        args.vocab_size, args.teacher_dim, seed=args.seed
    )

    # --- geometría ---
    vertices = generate_vertices(
        num_layers=args.num_layers,
        verts_per_layer=args.verts_per_layer,
        dim=args.layer_dim,
        seed=args.seed,
    )

    # --- modelo FLSE ---
    model = FLSEModel(
        vocab_size=args.vocab_size,
        vertices=vertices,
        teacher_dim=args.teacher_dim,
    )

    # --- entropías objetivo ---
    if args.target_entropies is None:
        # default: más difuso arriba, más concentrado abajo
        max_ent = torch.log(torch.tensor(args.verts_per_layer, dtype=torch.float))
        target_entropies = torch.linspace(
            max_ent, max_ent * 0.3, args.num_layers
        )
    else:
        target_entropies = torch.tensor(args.target_entropies, dtype=torch.float)

    print("Target entropies:", target_entropies.tolist())

    # --- entrenamiento ---
    history = train_distillation(
        model=model,
        teacher_vectors=teacher_vectors,
        target_entropies=target_entropies,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_ent=args.lambda_ent,
        lr=args.lr,
        device=args.device,
        log_every=50,
    )

    # --- diagnóstico rápido ---
    idx = torch.tensor([0, 1, 2])
    emb = model.flse_embedding(idx)
    proj = model(idx)

    print("FLSE embedding shape:", emb.shape)
    print("Projected (teacher) shape:", proj.shape)
    print("Última loss:", history["loss"][-1])


if __name__ == "__main__":
    main()
