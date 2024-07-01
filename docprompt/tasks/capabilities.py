from enum import Enum


class PageLevelCapabilities(str, Enum):
    """
    Represents a capability that a provider can fulfill
    """

    PAGE_RASTERIZATION = "page-rasterization"
    PAGE_LAYOUT_OCR = "page-layout-ocr"
    PAGE_TEXT_OCR = "page-text-ocr"
    PAGE_CLASSIFICATION = "page-classification"
    PAGE_MARKERIZATION = "page-markerization"
    PAGE_SEGMENTATION = "page-segmentation"
    PAGE_VQA = "page-vqa"
    PAGE_TABLE_IDENTIFICATION = "page-table-identification"
    PAGE_TABLE_EXTRACTION = "page-table-extraction"


class DocumentLevelCapabilities(str, Enum):
    DOCUMENT_VQA = "multi-page-document-vqa"
