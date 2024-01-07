from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field

from docprompt.schema.layout import TextBlock


class OPERATIONS(Enum):
    IMAGE_PROCESSING = "image_processing"
    LAYOUT_ANALYSIS = "layout_analysis"
    TEXT_EXTRACTION = "text_extraction"
