import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama

# Local modules
from document_parsing import make_default_pdf_options, DocumentParser
from chunk_embedding import ChunkEmbedder
from faiss_indexing import FaissIndex


class DocumentExplorer:
    def __init__(self, doc_folder: Path, config_path: Path):
        self.doc_folder = doc_folder
        self.config = self._load_config(config_path)
        self.embedder = None
        self.index = None
        self.llm = None
        self.prompt_template = ""
        self.top_k = 5

    def _load_config(self, path):
        """Load JSON configuration file."""

        if not path.is_file():
            logging.error(f"Config file not found: {path}")
            sys.exit(1)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {path}: {e}")
            sys.exit(1)

    def _validate_folder(self):
        """Ensure folder has acceptable number and types of files."""
        ingestion_cfg = self.config.get("ingestion", {})
        max_files = ingestion_cfg.get("max_files", 100)
        allowed_formats = ingestion_cfg.get("allowed_formats", ["pdf"])

        files = [p for p in self.doc_folder.iterdir() if p.is_file()]

        if not files:
            logging.error(
                f"No files found in {self.doc_folder}. Please add at least one file."
            )
            sys.exit(1)

        # too many files
        if len(files) > max_files:
            logging.error(
                f"Too many files in {self.doc_folder}: "
                f"found {len(files)}, but the maximum is {max_files}."
            )
            sys.exit(1)

        # invalid extensions
        invalid_files = [
            p for p in files if p.suffix.lower().lstrip(".") not in allowed_formats
        ]
        if invalid_files:
            names = ", ".join(p.name for p in invalid_files)
            logging.error(
                f"Unsupported file formats in {self.doc_folder}: "
                f"{len(invalid_files)} file(s) not allowed (allowed: {allowed_formats}): {names}"
            )
            sys.exit(1)

    def ingest_documents(self):
        print("Validating the folder...\n")
        self._validate_folder()

        print("Processing the documents...\n")
        # Initialize document parser options (docling) and embedder
        pipeline_opts = make_default_pdf_options()
        indexing_cfg = self.config.get("indexing", {})
        model_name = indexing_cfg.get("model_name", "bge-m3")
        vector_type = indexing_cfg.get("vector_type", "dense")
        self.embedder = ChunkEmbedder(model_name=model_name, vector_type=vector_type)

        # Convert PDFs → Markdown → Embeddings
        chunk_list = []
        for pdf_path in self.doc_folder.iterdir():
            parser = DocumentParser(str(pdf_path), pipeline_opts)
            md = parser.generate_clean_markdown(embed_images=False)
            chunks = [md]  # could replace this with a chunking strategy

            for chunk in chunks:
                embedding = self.embedder.embed_query(chunk)
                chunk_list.append([str(pdf_path), chunk, embedding])
                # chunk_list += chunks

        # Create dataframe of chunks
        chunks_df = pd.DataFrame(
            chunk_list, columns=["file_name", "text", "embeddings"]
        )

        # Indexing with FAISS
        print("Indexing the documents...\n")
        chunk_emb_matrix = np.vstack(chunks_df["embeddings"])
        index_type = indexing_cfg.get("index_type", "IndexFlatL2")
        self.index = FaissIndex(
            chunk_texts=chunks_df["text"],
            chunk_embeddings=chunk_emb_matrix,
            index_type=index_type,
        )

        # Load LLM for Q&A
        qa_cfg = self.config.get("query_answering", {})
        llm_name = qa_cfg.get("model_name", "deepseek-r1:8b")
        temperature = qa_cfg.get("temperature", 0)
        self.prompt_template = qa_cfg.get("prompt_template", "")
        self.top_k = qa_cfg.get("top_k", 5)
        supported_models = qa_cfg.get("supported_models", "None")

        print("Loading the LLM for question answering...\n")
        if llm_name == "deepseek-r1:8b":
            self.llm = ChatOllama(model=llm_name, temperature=temperature)
        else:
            f"Unsupported model: {llm_name}. Supported models are: {supported_models}"
            sys.exit(1)

    def answer_query(self, query: str) -> str:
        query_emb = self.embedder.embed_query(query)
        top_docs = self.index.retrieve_top_k_chunks(query_emb, top_k=self.top_k)
        prompt = self.prompt_template.format(context=top_docs, user_query=query)

        return self.llm.invoke(prompt).content.split("</think>")[-1]
