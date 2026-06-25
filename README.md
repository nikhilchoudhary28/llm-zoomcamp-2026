cat << 'EOF' > README.md
# LLM Zoomcamp 2026 - Module 2: Vector Search Homework

This repository contains the practical implementation and homework solutions for Module 2 of the LLM Zoomcamp. The focus of this module is on understanding the core mechanics of vector search, text search, and hybrid ranking algorithms from scratch.

## 🛠️ Tech Stack & Utilities
- **Environment & Dependency Management:** `uv` (fast Python package installer and resolver)
- **Embeddings:** ONNX Runtime (`Embedder` using `Xenova/all-MiniLM-L6-v2` for lightweight, CPU-efficient vector generation)
- **Data Source:** Pinned GitHub repository lessons data (commit `8c1834d`) parsed using `gitsource`
- **Search Engines:** Pure NumPy matrix operations (`X.dot(v)`) alongside evaluation comparisons with `minsearch`

## 🎯 Implementation Details
- **Q1: Embedding Queries:** Generating 384-dimensional dense vectors using a lightweight local ONNX workflow.
- **Q2 & Q3: Hand-Crafted Vector Search:** Manually parsing document chunks, calculating normalized dot products, and identifying highest-scoring lesson profiles.
- **Q4 & Q5: Vector vs. Keyword Evaluation:** Contrasting dense semantic retrieval against traditional text-matching indices.
- **Q6: Hybrid Search Engine:** Merging separate vector and text retrieval ranks using **Reciprocal Rank Fusion (RRF)** with a tuning constant of `k = 60`.
EOF