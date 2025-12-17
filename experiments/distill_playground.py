import argparse
from pathlib import Path
import numpy as np
import torch

from flse.geometry import generate_vertices
from flse.model import FLSEModel
from flse.training import train_distillation
from flse.losses import batch_entropy_per_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Playground rápido para distillation FLSE con distintos teachers y parámetros."
    )
    parser.add_argument("--teacher-path", type=Path, default=None,
                        help="Ruta a matriz teacher en .npy (shape: vocab, dim). Si no se pasa, se usa un teacher aleatorio.")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Tamaño de vocab. Si se carga un .npy se infiere; opcionalmente se puede truncar con este valor.")
    parser.add_argument("--teacher-dim", type=int, default=64,
                        help="Dimensión del embedding teacher (solo cuando se usa teacher aleatorio).")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--verts-per-layer", type=int, default=16)
    parser.add_argument("--dim", type=int, default=16, help="Dimensión interna por capa.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-ent", type=float, default=0.1,
                        help="Peso de la regularización de entropía.")
    parser.add_argument("--lambda-entropies", type=float, nargs="+", default=None,
                        help="Lista de pesos de entropía por capa (opcional). Debe coincidir con num_layers.")
    parser.add_argument("--logit-temps", type=float, nargs="+", default=None,
                        help="Temperaturas por capa para escalar los logits antes del softmax. Largo = num_layers.")
    parser.add_argument("--target-entropy", type=float, default=1.5,
                        help="Valor base de entropía por capa si no se pasa una lista.")
    parser.add_argument("--target-entropies", type=float, nargs="+", default=None,
                        help="Lista de entropías objetivo por capa; debe coincidir con num_layers.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, default=None,
                        help="Ruta para guardar el checkpoint del modelo (state_dict y config).")
    return parser.parse_args()


def prepare_teacher(args: argparse.Namespace) -> tuple[torch.Tensor, int, int]:
    if args.teacher_path:
        if not args.teacher_path.exists():
            raise FileNotFoundError(f"No encontré el archivo {args.teacher_path}")
        data = np.load(args.teacher_path)
        if data.ndim != 2:
            raise ValueError("El teacher debe ser una matriz 2D (vocab, dim).")
        teacher_vectors = torch.tensor(data, dtype=torch.float32)
        vocab_size = teacher_vectors.shape[0]
        teacher_dim = teacher_vectors.shape[1]
        if args.vocab_size:
            vocab_size = min(args.vocab_size, vocab_size)
            teacher_vectors = teacher_vectors[:vocab_size]
    else:
        if args.vocab_size is None:
            raise ValueError("Cuando no pasás --teacher-path, especificá --vocab-size.")
        torch.manual_seed(args.seed)
        teacher_vectors = torch.randn(args.vocab_size, args.teacher_dim)
        vocab_size = args.vocab_size
        teacher_dim = args.teacher_dim

    return teacher_vectors, vocab_size, teacher_dim


def build_target_entropies(args: argparse.Namespace, num_layers: int) -> torch.Tensor:
    if args.target_entropies is not None:
        entropies = torch.tensor(args.target_entropies, dtype=torch.float32)
        if entropies.numel() != num_layers:
            raise ValueError("target-entropies debe tener exactamente num_layers valores.")
        return entropies
    return torch.full((num_layers,), args.target_entropy, dtype=torch.float32)


def build_lambda_entropies(args: argparse.Namespace, num_layers: int) -> torch.Tensor | None:
    if args.lambda_entropies is None:
        return None
    weights = torch.tensor(args.lambda_entropies, dtype=torch.float32)
    if weights.numel() != num_layers:
        raise ValueError("lambda-entropies debe tener exactamente num_layers valores.")
    return weights


def build_logit_temps(args: argparse.Namespace, num_layers: int) -> torch.Tensor | None:
    if args.logit_temps is None:
        return None
    temps = torch.tensor(args.logit_temps, dtype=torch.float32)
    if temps.numel() != num_layers:
        raise ValueError("logit-temps debe tener exactamente num_layers valores.")
    return temps


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    teacher_vectors, vocab_size, teacher_dim = prepare_teacher(args)
    vertices = generate_vertices(
        num_layers=args.num_layers,
        verts_per_layer=args.verts_per_layer,
        dim=args.dim,
    )
    logit_temps = build_logit_temps(args, args.num_layers)
    model = FLSEModel(
        vocab_size=vocab_size,
        vertices=vertices,
        teacher_dim=teacher_dim,
        logit_temps=logit_temps,
    )

    targets = build_target_entropies(args, args.num_layers)
    lambda_ents = build_lambda_entropies(args, args.num_layers)
    device = None if args.device == "auto" else args.device

    print(f"Entrenando FLSE con vocab={vocab_size}, capas={args.num_layers}, "
          f"verts/capa={args.verts_per_layer}, dim capa={args.dim}, teacher_dim={teacher_dim}")
    model, _ = train_distillation(
        model=model,
        teacher_vectors=teacher_vectors,
        target_entropies=targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_ent=args.lambda_ent,
        lambda_entropies=lambda_ents,
        device=device,
    )

    sample = torch.arange(min(8, vocab_size), device=model.logits.device)
    with torch.no_grad():
        ent = batch_entropy_per_layer(model, sample)
        emb = model.flse_embedding(sample).cpu()

    print("\nEntropías promedio por capa en la muestra:", ent.cpu().tolist())
    print("Embedding fractal de la primera palabra (flatten):")
    print(emb[0].tolist())

    if args.save_path:
        ckpt = {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "num_layers": args.num_layers,
                "verts_per_layer": args.verts_per_layer,
                "dim": args.dim,
                "teacher_dim": teacher_dim,
                "target_entropies": targets.cpu().tolist(),
                "lambda_ent": args.lambda_ent,
                "lambda_entropies": None if lambda_ents is None else lambda_ents.cpu().tolist(),
                "logit_temps": None if logit_temps is None else logit_temps.cpu().tolist(),
            },
        }
        torch.save(ckpt, args.save_path)
        print(f"\nGuardado checkpoint en {args.save_path}")


if __name__ == "__main__":
    main()
