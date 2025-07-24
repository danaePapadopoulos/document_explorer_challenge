# PDF Question-Answering System

## 👤 Danae Papadopoulos

---

## 📌 Introduction

This project implements a command-line-based question-answering system where users can input a folder containing up to 100 PDF documents and ask questions related to their contents. The system processes the documents, extracts relevant information, and uses a local LLM to generate the most relevant and context-aware answers.


## 🚀 How to Run

### 1. Install Ollama

Ollama is used to run the local LLM model.

- Download from: [https://ollama.com/download](https://ollama.com/download)
- Or install via Homebrew (macOS only):

  ```bash
  brew install ollama
  ```

### 2. Pull the DeepSeek Model
Run the following command to download the required model:

  ```bash
  ollama pull deepseek-r1:8b
  ```
⚠️ Note: The model download may take 30 minutes to 1 hour depending on your network. The first-time startup may also require some extra time for initialization.

### 3. Set Up the Environment
Create and activate the Python environment using conda:

  ```bash
  conda env create -f environment.yml
  conda activate <your_env_name>
  ```

### 4. Run the CLI Application
Use the following command to run the project:

```bash
  python src/main.py --doc_folder <path_to_your_folder>
```
  Replace <path_to_your_folder> with the path to the folder containing your PDF documents.


## 🧠 Design Decisions, Technology Choices & Assumptions

### 📁 1. Folder Validation

The first step is to validate the input folder provided via CLI. We check:

- If the given path exists  
- If it contains any files  
- If all files are of the allowed format (PDF), and if their number is below the maximum limit  

These constraints are defined in the `config.json` under the `"ingestion"` section. Setting this logic in a config file supports future extensibility. For example, the system already supports other formats like `.docx` or `.xlsx`, and these could be enabled simply by updating the config file — though no guarantees are made about the quality of answers for non-PDFs.


### 🧹 2. Document Preprocessing

Next, the system preprocesses the documents to make them LLM-readable. Specifically, all valid files are converted to **Markdown** format to preserve the structure — such as titles, subtitles, paragraphs, etc.

This step is handled using the **Docling** library, which provides robust PDF parsing capabilities, including:

- Accurate page layout and reading order
- Table and code block detection
- Formula handling

An interesting (but unused) feature is that Docling also extracts base64 image URIs. While image understanding wasn’t implemented here, a future extension could send these to a **multimodal model** to generate image captions and embed them into the Markdown — giving the LLM better multimodal context. However, due to unknowns in the PDF content (e.g., whether they contain images) and the high runtime cost of image processing, this wasn’t included in the current version.


### 🔍 3. RAG-Based Retrieval Strategy

Even if we could fit all document content into the LLM’s context window, it’s unlikely the model would retain everything accurately. Instead, it’s more effective to **select only the most relevant documents based on the user query**.

To do this, the system implements a **simple RAG (Retrieval-Augmented Generation)** strategy:

- All documents are embedded using the **BGE-M3 model**  
- The user query is also embedded  
- We compute the similarity between the query and each document  
- We retrieve the top-K most relevant documents to feed into the LLM  

> 🔍 **Note**: Chunking (i.e., splitting documents into smaller parts) is not implemented here. While common in RAG systems, optimal chunking strategies vary (e.g., overlapping chunks, summarizing chunks), and the effectiveness depends heavily on the document types. Since we don’t know the documents ahead of time, this version uses full-document embeddings as a **baseline**. Future iterations could test chunking + evaluate impact using tools like **RAGAS**.


### 🧠 Why BGE-M3?

We use the **BGE-M3 embedding model** because:

- It performs well across dense, sparse, and multi-vector retrieval tasks  
- It’s a top performer among open-source models on common benchmarks  
- It supports flexible retrieval strategies (though only dense retrieval is used here)  

Another possible enhancement is to **combine sparse and dense retrieval** to improve recall — but this would again need evaluation to ensure it actually improves answer quality.


### ⚡ 4. Efficient Similarity Search with FAISS

To perform fast similarity search between query and document embeddings, we use **FAISS**:

- It supports large-scale vector search  
- Implements high-speed indexing algorithms, including GPU-accelerated ones  
- For this project, we use the `IndexFlatIP` algorithm (configurable via `config.json`)  

Flat indexes are sufficient here because we’re dealing with a small number of documents (max 100), so we avoid the complexity of IVF or PQ-based indexing.


### 🤖 5. Answer Generation with DeepSeek

Once the top relevant documents are selected, they are passed to the LLM to generate an answer.

The model used is `deepseek-r1:8b`, run locally via **Ollama**.

Reasons for this choice:

- It is open-source and free to use locally (no API costs)  
- Easily swappable via Ollama with models like LLaMA 3 or Mistral  
- Has a **large context window (128K tokens)** while being relatively lightweight (~5.2 GB)  
- Shows strong reasoning ability, helpful for queries that require document understanding rather than surface-level answers  

>⏳ While it may take slightly longer to generate a response compared to smaller models, response quality and relevance were prioritized in this project — especially since the challenge emphasizes output accuracy over speed.

> 🔐 Hallucination control is currently handled via **prompt engineering**. A future version could implement LLM-based self-verification (e.g., "judge" models) to reduce hallucinations further.

---

### 📈 Future Directions

- Implement document **chunking strategies** and test with RAGAS to assess quality  
- Add **multimodal support** (image captioning for PDFs with visuals)  
- Explore **hybrid retrieval** (dense + sparse) to improve relevance  
- Extend to handle larger corpora using persistent vector DBs  
- Implement feedback loops or reranking using answer quality  

---

## 🗂️ Code Overview

This section provides a brief description of each main source file and its role in the system.

### 📄 `main.py`  
Implements the CLI entry point that initializes the system, ingests the document folder, and enters an interactive Q&A loop. It handles argument parsing, basic error checking, and delegates all core logic to the `DocumentExplorer`.

### 📄 `document_explorer.py`  
Coordinates the full pipeline: it validates the input folder, converts documents to Markdown, embeds them, indexes them with FAISS, and loads the LLM for answering queries. The `DocumentExplorer` class encapsulates ingestion, indexing, and prompt-driven response generation using local models.

### 📄 `document_parsing.py`  
Handles PDF-to-Markdown conversion using Docling with support for OCR, tables, and optional image embedding. Provides a `DocumentParser` class that memoizes results and ensures clean, structured output for LLM processing.

### 📄 `chunk_embedding.py`  
Implements a `ChunkEmbedder` class that generates dense embeddings using the **BGE-M3** model for queries or text chunks. Supports lazy loading and prepares vectors for similarity search, with placeholders for future support of sparse embeddings and additional models.

### 📄 `faiss_indexing.py`  
Defines a `FaissIndex` class that builds and queries FAISS indexes (Flat, IVF, or IVFPQ) for fast vector similarity search. Supports top-k retrieval of the most relevant text chunks based on dense embeddings, with configurable indexing strategies and scoring output.

### Summary

The current implementation provides a **fully local, configurable, and extendable** QA system that works out-of-the-box on folders of PDFs. It serves as a baseline and foundation for more advanced RAG setups, chunking logic, multimodal support, and intelligent reranking.
