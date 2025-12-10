# FLSE — Fractal Layered Spherical Embeddings

FLSE es un modelo experimental de *embeddings fractales jerárquicos* basado en:

- capas semánticas múltiples,
- geometría esférica en alta dimensión,
- representación fractal del significado,
- distilación desde embeddings teacher (GloVe, fastText, LaBSE, etc.),
- control explícito de polisemia y granularidad,
- entropía por capa como regulador estructural.

El objetivo del proyecto es explorar una alternativa teórica y práctica a los
embeddings tradicionales (GloVe / word2vec / fastText / BERT embeddings),
desacoplando:

1. significado macro (conceptos generales),
2. significado meso (categorías y subcategorías estables),
3. significado micro (polisemia contextual, jergas, dominio específico),

mediante un *espacio fractal* compuesto por múltiples hiperesferas conectadas.

---

## ✨ Motivación

Los modelos modernos de lenguaje usan grandes espacios vectoriales altamente
entrelazados donde:

- nociones generales,
- conceptos específicos,
- jergas contextuales,
- relaciones sintácticas

se mezclan en un único embedding difícil de interpretar.

FLSE propone una descomposición estructurada:

- cada palabra se representa mediante un vector compuesto (una capa = una esfera),
- cada capa escoge una combinación suave de vértices,
- la combinación se regula con una entropía objetivo (capa alta = distribuciones amplias, capa baja = distribuciones concentradas),
- el embedding final es la concatenación de todas las capas.

Esto permite:

- interpretar significado según escala,
- manejar polisemia explícita,
- robustez a spanglish, jergas y mezclas culturales,
- especialización progresiva sin interferir con capas superiores,
- integración sencilla con embeddings teacher multilingües.

---

## 📐 Arquitectura

```
Capa 1 (macro)       → vértices esféricos, semántica general
input → Capa 2 (meso)→ categorías y subcategorías
Capa 3 (micro)       → jergas, uso específico, dominios
...
Capa N (fina)        → detalles contextuales, sintaxis opcional

Embeddings finales = concat(capa1, capa2, ... capaN)
```

Cada capa tiene:

- `V` vértices distribuidos sobre una hiperesfera `D`–dimensional
- para cada palabra se aprenden `V` logits
- se aplica `softmax` → pesos por vértice
- el embedding de capa es la mezcla convexa de sus vértices

---

## 🧠 Entrenamiento

FLSE se entrena mediante **distillation** desde un embedding teacher:

- GloVe (inglés)
- fastText (multilingüe)
- LaBSE (multilingüe alineado)
- SBERT multilingual

La pérdida incluye:

1. MSE entre FLSE y el embedding teacher
2. Regularización de entropía por capa

```
loss = MSE(FLSE, teacher) + λ * (Entropy_per_layer - TargetEntropy)²
```

Esto fuerza a cada capa a aprender un nivel controlado de granularidad, estabilizando el espacio fractal.

---

## 🚀 Getting Started

### Instalación

Clonar el repositorio:

```bash
git clone https://github.com/santiagoNieva/FLSE.git
cd FLSE
```

### Usando Poetry

```bash
poetry install
```

Activar entorno:

```bash
source $(poetry env info --path)/bin/activate
```

### Ejecutar tests

```bash
make test
```

---

## 🧠 Uso básico

```python
import torch
from flse.geometry import generate_vertices
from flse.model import FLSEModel

vertices = generate_vertices(num_layers=3, verts_per_layer=16, dim=16)
model = FLSEModel(vocab_size=1000, vertices=vertices, teacher_dim=64)

embedding = model.flse_embedding(torch.tensor(42))
print(embedding.shape)
```

---

## ☁️ FLSE en Google Colab

```python
!git clone https://github.com/santiagoNieva/FLSE.git
%cd FLSE
!pip install -e .
```

Importar:

```python
from flse.model import FLSEModel
from flse.geometry import generate_vertices
```

---

## 🧪 Playground rápido con un teacher

Hay un script sencillo para jugar con parámetros y un embedding teacher (propio o
aleatorio):

```bash
# Teacher aleatorio para smoke-test rápido
python experiments/distill_playground.py --vocab-size 500 --num-layers 3 --verts-per-layer 16 --dim 16 --teacher-dim 64 --epochs 3 --target-entropy 1.2

# Usando un teacher guardado en .npy (shape: vocab, dim)
python experiments/distill_playground.py --teacher-path data/teacher.npy --num-layers 4 --verts-per-layer 24 --dim 16 --epochs 5 --lambda-ent 0.2 --target-entropies 1.5 1.0 0.8 0.5
```

Tips:

- `--device auto|cpu|cuda` elige la aceleración (auto usa CUDA si está disponible).
- Las entropías objetivo pueden ser una sola (`--target-entropy`) o una lista
  por capa (`--target-entropies ...`). Sirven para controlar la mezcla suave de
  vértices en cada nivel.
- El teacher se carga desde un `.npy` de shape `(vocab, dim)`. Podés recortar con
  `--vocab-size` si querés probar solo un subconjunto.

---

## 📜 Licencia

Este proyecto se publica bajo **Creative Commons Attribution–NonCommercial 4.0 (CC BY-NC 4.0)**.

Esto implica:

- podés usar el código para fines personales, académicos o experimentales,
- podés modificar y redistribuir derivaciones bajo la misma licencia,
- NO está permitido el uso comercial sin autorización explícita del autor,
- empresas o instituciones deben solicitar una licencia comercial.

La licencia podrá revisarse cuando el proyecto alcance mayor madurez.

---
