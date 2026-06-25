# LLM Zoomcamp 2026 - Portfolio & Homework Solutions

This repository contains my practical implementations, codebase, and homework solutions for the LLM Zoomcamp course.

## 📁 Repository Structure
- `homework/module_01.ipynb`: RAG intro, basic tokenization, and search scoring logic.
- `homework/homework_module_2.ipynb`: Complete implementation of dense vector and hybrid search pipelines.
- `download.py` / `embedder.py`: Utilities for managing lightweight ONNX text embedding runtimes.

---

## 🛠️ Tech Stack & Utilities
- **Environment & Dependency Management:** `uv` (fast, lightweight Python package installer and workspace resolver).
- **Embeddings Runtime:** ONNX Runtime (`Embedder` utilizing `Xenova/all-MiniLM-L6-v2` for CPU-efficient vector generation without heavy PyTorch layers).
- **Search Indices:** Pure NumPy matrix computations (`X.dot(v)`) combined with evaluations using the `minsearch` utility.
- **Data Gathering:** Automated ingestion pipelines parsing multi-module markdown documents using the `gitsource` driver.

---

## 🎯 Implementation Highlights

### 🔹 Module 1: Introduction to Search & RAG
- **Knowledge Base Setup:** Parsed unstructured data files to form a searchable document index.
- **Basic Retrieval:** Configured exact keyword search criteria and initial text-matching scores.
- **RAG Execution:** Built out basic orchestration prompts for generative modeling tasks.

### 🔹 Module 2: Advanced Vector Search & Ranking Fusion
- **Dense Vector Embedding:** Implemented 384-dimensional dense semantic array mapping for cross-document query handling.
- **Semantic Retrieval:** Built a custom vector space retrieval mechanism from scratch using normalized cosine dot products.
- **Hybrid Search Engine:** Combined the precision of classic exact-match keyword search with the conceptual awareness of vector search using **Reciprocal Rank Fusion (RRF)** at a tuning constant of `k = 60`.