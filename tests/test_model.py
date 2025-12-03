from loguru import logger
import torch
from flse.geometry import generate_vertices
from flse.model import FLSEModel

def test_flse_model_forward_and_embedding():
    vocab_size = 10
    num_layers = 2
    verts_per_layer = 8
    dim = 4
    teacher_dim = 6

    logger.info("Inicializando modelo FLSE para pruebas...")
    logger.info(f"vocab_size={vocab_size}, num_layers={num_layers}, "
                f"verts_per_layer={verts_per_layer}, dim={dim}, teacher_dim={teacher_dim}")

    vertices = generate_vertices(num_layers, verts_per_layer, dim)
    model = FLSEModel(vocab_size=vocab_size, vertices=vertices, teacher_dim=teacher_dim)

    idx_batch = torch.tensor([0, 3, 5])
    logger.info(f"Probando forward con batch: {idx_batch.tolist()}")

    out = model(idx_batch)
    logger.info(f"Output shape: {out.shape}")

    assert out.shape == (3, teacher_dim)

    emb = model.flse_embedding(torch.tensor(2))
    logger.info(f"Embedding simple (shape={emb.shape}): {emb.tolist()}")

    assert emb.shape == (num_layers * dim,)

    idx_multi = torch.tensor([1, 4, 7])
    emb_batch = model.flse_embedding(idx_multi)
    logger.info(f"Embedding batch (shape={emb_batch.shape}): idx={idx_multi.tolist()}")

    assert emb_batch.shape == (3, num_layers * dim)
