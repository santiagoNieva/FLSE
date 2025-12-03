# FLSE — Fractal Layered Spherical Embeddings

FLSE es un modelo experimental de *embeddings fractales jerárquicos* basado en:

- **capas semánticas múltiples**,  
- **geometría esférica en alta dimensión**,  
- **representación fractal del significado**,  
- **distilación desde embeddings teacher** (GloVe, fastText, LaBSE, etc.),  
- **control explícito de polisemia y granularidad**,  
- **entropía por capa como regulador estructural**.

El objetivo del proyecto es explorar una alternativa teórica y práctica a los
embeddings tradicionales (GloVe / word2vec / fastText / BERT embeddings),
desacoplando:

1. **significado macro** (conceptos generales),  
2. **significado meso** (categorías y subcategorías estables),  
3. **significado micro** (polisemia contextual, jergas, dominio específico),  

mediante un **espacio fractal** compuesto por múltiples hiperesferas conectadas.

---

## ✨ Motivación

Los modelos modernos de lenguaje usan grandes espacios vectoriales altamente
entrelazados donde:

- las nociones generales,  
- las específicas,  
- las jergas contextuales,  
- y las relaciones sintácticas  

se mezclan en un único embedding difícil de interpretar.

FLSE propone una descomposición estructurada:

- cada palabra se representa mediante un **vector compuesto** (una capa = una esfera),
- cada capa escoge una **combination suave de vértices**,  
- la combinación se regula con una **entropía objetivo** (capa alta = distribuciones amplias, capa baja = distribuciones concentradas),
- el embedding final es la **concatenación de todas las capas**.

Esto permite:

- interpretación por escala semántica,
- polisemia explícita,
- robustez a *spanglish*, jergas y mezclas culturales,
- especialización progresiva sin interferir con capas superiores,
- integración sencilla con modelos teacher multilingües.

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

FLSE se entrena por **distillation** desde un embedding teacher:

- GloVe (inglés)
- fastText (multilingüe, español incluido)
- LaBSE (multilingüe alineado)
- SBERT multilingual, etc.

La pérdida incluye:

1. **MSE** entre FLSE y el teacher  
2. **Regularización de entropía por capa**

```
loss = MSE(FLSE, teacher) + λ * (Entropy_per_layer - TargetEntropy)²
```

---

## 📜 Licencia

Este proyecto se publica bajo **Creative Commons Attribution–NonCommercial 4.0 (CC BY-NC 4.0)**.

Esto significa:

- podés leer, estudiar y modificar el código,
- podés usarlo con fines académicos, personales o experimentales,
- **NO** está permitido el uso comercial sin autorización expresa del autor,
- las empresas o instituciones que deseen integrarlo en productos deberán solicitar una licencia comercial.

Este esquema es temporal durante la etapa de investigación del proyecto.
