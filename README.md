# PDF Question-Answering System

## 👤 Author
**Danae Papadopoulos**

---

## 📌 Introduction

This project implements a command-line-based question-answering system that enables users to input a folder of PDF documents and ask questions related to the content of those documents. The system processes the documents, extracts relevant information, and uses a local LLM to respond with the most pertinent answers.

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
  brew install ollama
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
  Replace <path_to_your_folder> with the path to the folder containing your PDF documents.
  ```
