"""
CLI tool for ingesting documents into the RAG system.
Usage:
    python ingest_cli.py <file_path> [--category <category>]
    python ingest_cli.py <folder_path> [--category <category>]

Examples:
    python ingest_cli.py ./docs/placement_rules.pdf --category placements
    python ingest_cli.py ./docs/ --category academics
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.rag_pipeline import RAGPipeline
from config import CATEGORIES

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx"}


def ingest_file(pipeline: RAGPipeline, file_path: str, category: str):
    """Ingest a single file."""
    print(f"\n📄 Processing: {Path(file_path).name}")
    print(f"   Category:  {category}")
    try:
        result = pipeline.ingest_document(file_path, category)
        print(f"   ✅ Success: {result['chunks_created']} chunks in {result['time_seconds']}s")
        return True
    except Exception as e:
        print(f"   ❌ Failed:  {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the College RAG system",
    )
    parser.add_argument(
        "path",
        help="Path to a file or folder to ingest",
    )
    parser.add_argument(
        "--category", "-c",
        choices=CATEGORIES,
        default="general",
        help=f"Document category (choices: {', '.join(CATEGORIES)})",
    )
    args = parser.parse_args()

    target = Path(args.path).resolve()
    if not target.exists():
        print(f"❌ Path not found: {target}")
        sys.exit(1)

    pipeline = RAGPipeline()

    # Collect files
    if target.is_file():
        files = [target]
    else:
        files = sorted(
            f for f in target.rglob("*")
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    if not files:
        print(f"❌ No supported files found ({', '.join(SUPPORTED_EXTENSIONS)})")
        sys.exit(1)

    print(f"🔍 Found {len(files)} file(s) to ingest")

    success = 0
    failed = 0
    for f in files:
        if ingest_file(pipeline, str(f), args.category):
            success += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Ingestion Summary")
    print(f"   Total:   {len(files)}")
    print(f"   Success: {success}")
    print(f"   Failed:  {failed}")
    print(f"   Store:   {pipeline.get_total_chunks()} total chunks")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
