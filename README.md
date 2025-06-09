# RAG4Thesis: Author-Aware Retrieval-Augmented QA for Academic Theses

This is a course side project implementing a simple **author-aware question-answering system** based on Retrieval-Augmented Generation (RAG). It is designed for use in an academic lab where students often need to search through multiple senior theses to extract specific experimental details.

The system retrieves relevant content **separately for each author**, and generates answer outputs grouped by author, using a **local LLM (Mistral-NeMo via Ollama)**. A pointwise reranking method is also included to improve answer quality.

> 📌 **Note:** This is a lightweight prototype for a class project, not a production-ready tool.

---

## 💡 Project Structure

### `rag_embedding.py`
- Loads PDFs from `./docs`
- Extracts text, splits into chunks, and embeds using HuggingFace model (`BAAI/bge-small-en-v1.5`)
- Saves all vectors into a Chroma vector store with metadata (`author`, `source`)

### `rag_retrieval.py`
- Baseline retrieval: Retrieves top-3 chunks *per author* based on cosine similarity
- Generates answers using **Mistral-NeMo via Ollama**

### `rag_retrieval_pointwise.py`
- Adds **pointwise reranking**: Retrieves top-10 chunks *per author*, scores each with the LLM, and reranks
- Generates more accurate answers based on the top-3 scored passages

---

## 🧠 Key Features

- ✅ Author-aware retrieval using metadata filtering (`filter={"author": author}`)
- ✅ Local inference using **Ollama + Mistral-NeMo**
- ✅ Support for pointwise reranking using LLM scoring
- 📄 Manual QA dataset with 5 domain-specific questions for basic evaluation

---

## 🛠 How to Use

1. Place your thesis PDFs in `./docs`
2. Run the embedding script:
   ```bash
   python rag_embedding.py
3. Start answering questions (choose either script):
   ```bash
   python rag_retrieval.py              # baseline
   python rag_retrieval_pointwise.py    # with pointwise reranking

---

## 📊 Sample Output

**Query:**  
What 3D printing techniques did they use in their research?

**Output Format:**  
The system retrieves and generates answers **separately by author**, like the example below:

![Sample output image](./ml_output.png)




