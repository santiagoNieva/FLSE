from loguru import logger
import torch
from flse.geometry import generate_vertices

def test_generate_vertices_shape_and_norm():
    num_layers = 3
    verts_per_layer = 16
    dim = 8

    logger.info("Generando vértices con parámetros: "
                f"num_layers={num_layers}, verts_per_layer={verts_per_layer}, dim={dim}")

    vertices = generate_vertices(num_layers, verts_per_layer, dim)
    logger.info(f"Shape obtenido: {vertices.shape}")

    norms = torch.linalg.norm(vertices, dim=-1)
    logger.info(f"Normas de la primera capa: {norms[0].tolist()}")

    assert vertices.shape == (num_layers, verts_per_layer, dim)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
