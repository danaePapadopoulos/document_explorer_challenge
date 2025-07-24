import logging
import time
from pathlib import Path
from typing import Any, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

logger = logging.getLogger(__name__)


def make_default_pdf_options() -> PdfPipelineOptions:
    """
    Create and return a default configuration for Docling PDF pipeline options.
    """

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    return pipeline_options


class DocumentParser:
    """
    Wrapper around Docling's document conversion pipeline.
    Takes a source document (e.g. PDF) and converts it to
    a structured format, with support for Markdown export.
    (And option to embed images)
    """

    def __init__(
        self,
        source_path: str,
        pipeline_opts: Optional[PdfPipelineOptions] = None,
    ):
        """
        Initialize a DocumentParser for the given source file.

        Parameters
        ----------
        source_path : str
            Path to the input document (PDF or compatible format).
        pipeline_opts : Optional[PdfPipelineOptions]
            Preconfigured pipeline options. If None, defaults will be used.
        """
        self.source_path = Path(source_path)
        self.pipeline_opts = pipeline_opts or make_default_pdf_options()
        self._converted: Optional[Any] = None  # cache for converted result

    def convert(self) -> Any:
        """
        Perform the actual document conversion using Docling.

        This method is memoized: repeated calls reuse the cached result.

        Returns
        -------
        Any
            The result of Docling's `convert()` method, including structured data.
        """
        if self._converted is None:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.pipeline_opts
                    )
                }
            )

            start = time.time()
            self._converted = converter.convert(str(self.source_path))
            elapsed = time.time() - start

            logger.info(f"Converted {self.source_path!r} in {elapsed:.2f}s")

        return self._converted

    def to_markdown(self, embed_images: bool = False) -> str:
        """
        Export the converted document into Markdown format.

        Parameters
        ----------
        embed_images : bool, optional
            Whether to embed images directly (base64) or reference them externally.

        Returns
        -------
        str
            Markdown string representing the document.
        """
        doc = self.convert().document
        mode = ImageRefMode.EMBEDDED if embed_images else ImageRefMode.REFERENCED

        return doc.export_to_markdown(image_mode=mode)

    def generate_clean_markdown(self, embed_images: bool = False) -> str:
        """
        Full pipeline: convert document and return Markdown output.

        Parameters
        ----------
        embed_images : bool, optional
            Whether to embed images in the Markdown or use file references.

        Returns
        -------
        str
            Clean Markdown string representing the document.
        """
        logger.info(
            "Generating clean Markdown (embed_images=%s) for %r",
            embed_images,
            self.source_path,
        )

        markdown = self.to_markdown(embed_images=embed_images)

        # Optional: Add post-processing steps here, such as embedding
        # images directly into the text (base64). This can enable
        # multimodal LLMs to generate image descriptions.
        # e.g., markdown = self._postprocess_markdown(markdown)

        return markdown
