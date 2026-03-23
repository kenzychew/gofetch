"""Document loading from PDF and text files."""

from pathlib import Path

from pypdf import PdfReader

from src.exceptions import IngestionError
from src.logging import get_logger
from src.schemas import Document

logger = get_logger(__name__)


def load_pdf(file_path: Path) -> Document:
    """Load a PDF file and extract its text content.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Document with extracted text and source metadata.

    Raises:
        IngestionError: If the PDF cannot be read or has no extractable text.
    """
    logger.info("Loading PDF", file_path=str(file_path))
    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        raise IngestionError(f"Failed to read PDF: {file_path}") from exc

    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    if not pages:
        raise IngestionError(f"No extractable text found in PDF: {file_path}")

    content = "\n\n".join(pages)
    logger.info("Loaded PDF", file_path=str(file_path), pages=len(pages), chars=len(content))
    return Document(content=content, source=file_path.name)


def load_text(file_path: Path) -> Document:
    """Load a plain text file.

    Args:
        file_path: Path to the text file.

    Returns:
        Document with file content and source metadata.

    Raises:
        IngestionError: If the file cannot be read.
    """
    logger.info("Loading text file", file_path=str(file_path))
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise IngestionError(f"Failed to read text file: {file_path}") from exc

    if not content.strip():
        raise IngestionError(f"Empty text file: {file_path}")

    logger.info("Loaded text file", file_path=str(file_path), chars=len(content))
    return Document(content=content, source=file_path.name)


LOADER_MAP = {
    ".pdf": load_pdf,
    ".txt": load_text,
}


def load_documents(data_dir: Path) -> list[Document]:
    """Load all supported documents from a directory.

    Scans the directory for .pdf and .txt files and loads them.

    Args:
        data_dir: Path to the directory containing documents.

    Returns:
        List of loaded documents.

    Raises:
        IngestionError: If the directory does not exist or contains no documents.
    """
    if not data_dir.exists():
        raise IngestionError(f"Data directory does not exist: {data_dir}")

    documents: list[Document] = []
    supported_extensions = set(LOADER_MAP.keys())

    for file_path in sorted(data_dir.iterdir()):
        if file_path.suffix.lower() in supported_extensions:
            loader = LOADER_MAP[file_path.suffix.lower()]
            try:
                doc = loader(file_path)
                documents.append(doc)
            except IngestionError:
                logger.warning("Skipping file due to load error", file_path=str(file_path))

    logger.info("Loaded documents", count=len(documents), data_dir=str(data_dir))
    return documents
