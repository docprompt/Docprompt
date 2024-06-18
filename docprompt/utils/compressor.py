import tempfile
from typing import Literal

from docprompt._exec.ghostscript import compress_pdf_to_bytes


def compress_pdf_bytes(
    file_bytes: bytes, *, compression: Literal["jpeg", "lossless"] = "jpeg"
) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()

        return compress_pdf_to_bytes(temp_file.name, compression=compression)
