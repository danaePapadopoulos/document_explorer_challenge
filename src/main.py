import argparse
import logging
import sys
from pathlib import Path

from document_explorer import DocumentExplorer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest PDFs and answer questions interactively"
    )

    parser.add_argument(
        "--doc_folder",
        type=str,
        required=True,
        help="Path to folder containing PDF files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    doc_folder = Path(args.doc_folder)

    if not doc_folder.is_dir():
        logging.error(f"Folder not found: {doc_folder}")
        sys.exit(1)

    config_path = Path(__file__).parent / "../config.json"
    explorer = DocumentExplorer(doc_folder, config_path)
    explorer.ingest_documents()

    print("\nReady! Ask your questions about the documents.")
    print("Type 'exit', 'quit' or Ctrl+C to exit.\n")

    try:
        while True:
            query = input(">  ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            answer = explorer.answer_query(query)
            print("\n" + answer + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
