"""
pdf_loader.py — PDF Document Processing Module

Handles PDF text extraction and intelligent chunking for downstream
retrieval and question generation.
"""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass, field

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class PDFProcessingResult:
    """Container for PDF processing output."""

    raw_text: str
    chunks: List[Document]
    page_count: int
    metadata: dict = field(default_factory=dict)


class PDFLoader:
    """
    Loads a PDF file, extracts text page-by-page, and splits it into
    overlapping chunks suitable for embedding and retrieval.

    Parameters
    ----------
    chunk_size : int
        Target number of characters per chunk.
    chunk_overlap : int
        Character overlap between consecutive chunks to preserve context.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_pdf(self, file_path: str) -> PDFProcessingResult:
        """
        Load a PDF and return extracted text plus chunks.

        Parameters
        ----------
        file_path : str
            Path to the PDF file on disk.

        Returns
        -------
        PDFProcessingResult
            Contains raw text, chunked documents, page count, and metadata.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a PDF or contains no extractable text.
        """
        # --- Validate input ---
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise ValueError("File must be a PDF (.pdf extension required)")

        logger.info("Loading PDF: %s", file_path)

        # --- Extract pages ---
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
        except Exception as exc:
            raise ValueError(f"Failed to read PDF: {exc}") from exc

        if not pages:
            raise ValueError("PDF contains no extractable text (may be scanned/image-only)")

        # --- Combine raw text and enrich metadata ---
        raw_text = "\n\n".join(page.page_content for page in pages)
        raw_text = raw_text.strip()

        if not raw_text:
            raise ValueError("PDF text extraction returned empty content")

        for i, page in enumerate(pages):
            page.metadata.update({
                "source": os.path.basename(file_path),
                "page": i + 1,
                "total_pages": len(pages),
            })

        # --- Chunk ---
        chunks = self.text_splitter.split_documents(pages)
        logger.info(
            "Extracted %d pages → %d chunks (size=%d, overlap=%d)",
            len(pages), len(chunks), self.chunk_size, self.chunk_overlap,
        )

        return PDFProcessingResult(
            raw_text=raw_text,
            chunks=chunks,
            page_count=len(pages),
            metadata={
                "file_name": os.path.basename(file_path),
                "file_size_bytes": os.path.getsize(file_path),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "total_chunks": len(chunks),
            },
        )

    def load_pdf_from_bytes(
        self, pdf_bytes: bytes, temp_dir: Optional[str] = None
    ) -> PDFProcessingResult:
        """
        Convenience wrapper: write bytes to a temp file, process, clean up.

        Useful when receiving uploads from Gradio where the file may arrive
        as raw bytes rather than a stable path.
        """
        import tempfile

        temp_dir = temp_dir or tempfile.gettempdir()
        tmp_path = os.path.join(temp_dir, "upload.pdf")

        try:
            with open(tmp_path, "wb") as f:
                f.write(pdf_bytes)
            return self.load_pdf(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
