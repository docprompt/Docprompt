import base64
import io
import os
import tempfile
import threading
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Tuple, Union

import fsspec
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from PIL import Image

from docprompt.rasterize import AspectRatioRule, ResizeModes, process_raster_image
from docprompt.schema.document import PdfDocument

TEMP_DIR_PREFIX = "raster_cache_"

if TYPE_CHECKING:
    from docprompt.schema.pipeline.node import PageNode
    from docprompt.schema.pipeline.node.document import DocumentNode


DEFAULT_CACHE_KEY = "default"


class Cache(ABC):
    """Abstract base class for caching mechanisms."""

    @abstractmethod
    def has_key(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass

    @abstractmethod
    def list_prefix(self, prefix: str) -> List[str]:
        """List all keys that start with the given prefix."""
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve a cached item by key."""
        pass

    @abstractmethod
    def set(self, key: str, data: bytes) -> None:
        """Cache an item with the given key."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached items."""
        pass

    @abstractmethod
    def pop(self, key: str, default: Optional[bytes] = None) -> Optional[bytes]:
        """Remove and return a cached item by key."""
        pass


class FilesystemCache(Cache):
    def __init__(self, cache_url: str, **fs_kwargs):
        """
        Initialize the filesystem cache.

        Args:
            cache_url (Optional[str]): The fsspec URL for the cache. Defaults to a temporary directory.
            cache_dir (Optional[str]): The directory path within the filesystem to store cache files.
        """
        self.cache_url = cache_url
        self._initalized = False

        protocol, _, path = cache_url.partition("://")

        if not path.endswith("/"):
            path += "/"

        if not protocol or not path:
            raise ValueError("Invalid cache URL provided.")

        if protocol == "temp":
            fs_tempdir = tempfile.gettempdir()

            base_fs = LocalFileSystem(auto_mkdir=True, **fs_kwargs)

            self.cache_dir = os.path.join(fs_tempdir, TEMP_DIR_PREFIX, path)
            self.fs = DirFileSystem(
                path=self.cache_dir, fs=LocalFileSystem(auto_mkdir=True)
            )
            self._finalizer = weakref.finalize(self, self._cleanup)
        else:
            base_fs = fsspec.filesystem(protocol, **fs_kwargs)

            self.cache_dir = path
            self.fs = DirFileSystem(path=self.cache_dir, fs=base_fs)

        self.lock = threading.RLock()
        self._initalize()

    def _initalize(self):
        with self.lock:
            if not self._initalized:
                self.fs.makedirs("", exist_ok=True)
                self._initalized = True

    def _cleanup(self):
        """Cleanup method to delete the temporary cache directory."""
        try:
            self.fs.rm("", recursive=True)
        except Exception:
            # Optionally log the exception
            pass  # Suppress exceptions to avoid issues during garbage collection

    def _get_path(self, key: str) -> str:
        """Generate a filesystem path for a given cache key."""
        return os.path.normpath(self.fs.sep.join([self.cache_dir, key]))

    def _create_dir(self, path: str) -> None:
        """Create a directory if it doesn't exist."""
        dir_path = os.path.dirname(path)
        relative_dir = os.path.relpath(dir_path, self.cache_dir)
        if relative_dir and not self.fs.exists(relative_dir):
            self.fs.makedirs(relative_dir, exist_ok=True)

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        path = self._get_path(key)
        return self.fs.exists(path)

    def list_prefix(self, prefix: str) -> List[str]:
        """List all keys that start with the given prefix."""
        with self.lock:
            search_path = self._get_path(prefix)
            if not search_path.endswith(self.fs.sep):
                search_path += self.fs.sep
            return self.fs.glob(f"{search_path}*")

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve a cached item by key."""
        path = self._get_path(key)
        if self.fs.exists(path):
            with self.lock:
                with self.fs.open(path, "rb") as f:
                    return f.read()

        return None

    def set(self, key: str, data: bytes) -> None:
        """Cache an item with the given key."""
        path = self._get_path(key)
        self._create_dir(path)

        with self.lock:
            with self.fs.open(path, "wb") as f:
                f.write(data)

    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            if self.fs.exists(""):
                self.fs.rm("", recursive=True)
                self.fs.makedirs("", exist_ok=True)

    def pop(self, key: str, default: Optional[bytes] = None) -> Optional[bytes]:
        """Remove and return a cached item by key."""
        path = self._get_path(key)
        with self.lock:
            if self.fs.exists(path):
                data = self.get(key)
                self.fs.rm(path)
                return data
        return default


class InMemoryCache(Cache):
    def __init__(self):
        """Initialize the in-memory cache."""
        self.store: Dict[str, bytes] = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve a cached item by key."""
        with self.lock:
            return self.store.get(key)

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        with self.lock:
            return key in self.store

    def list_prefix(self, prefix: str) -> List[str]:
        """List all keys that start with the given prefix."""
        with self.lock:
            return [key for key in self.store.keys() if key.startswith(prefix)]

    def set(self, key: str, data: bytes) -> None:
        """Cache an item with the given key."""
        with self.lock:
            self.store[key] = data

    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.store.clear()

    def pop(self, key: str, default: Optional[bytes] = None) -> Optional[bytes]:
        """Remove and return a cached item by key."""
        with self.lock:
            return self.store.pop(key, default)


def process_bytes(
    rastered: bytes,
    *,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    resize_mode: ResizeModes = "thumbnail",
    aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
    do_convert: bool = False,
    image_convert_mode: str = "L",
    do_quantize: bool = False,
    quantize_color_count: int = 8,
    max_file_size_bytes: Optional[int] = None,
) -> bytes:
    """
    Process rastered bytes with additional image processing.

    Args:
        rastered (bytes): The rastered image bytes.
        resize_width (Optional[int]): The width to resize to.
        resize_height (Optional[int]): The height to resize to.
        resize_mode (ResizeModes): The mode to resize the image.
        aspect_ratios (Optional[Iterable[AspectRatioRule]]): Aspect ratios to maintain.
        do_convert (bool): Whether to convert the image mode.
        image_convert_mode (str): The mode to convert the image to.
        do_quantize (bool): Whether to quantize the image colors.
        quantize_color_count (int): Number of colors for quantization.
        max_file_size_bytes (Optional[int]): Maximum file size in bytes.

    Returns:
        bytes: The processed rastered image bytes.
    """
    return process_raster_image(
        rastered,
        resize_width=resize_width,
        resize_height=resize_height,
        resize_mode=resize_mode,
        resize_aspect_ratios=aspect_ratios,
        do_convert=do_convert,
        image_convert_mode=image_convert_mode,
        do_quantize=do_quantize,
        quantize_color_count=quantize_color_count,
        max_file_size_bytes=max_file_size_bytes,
    )


class DocumentRasterCache:
    def __init__(
        self,
        document: PdfDocument,
        cache_url: Optional[str] = None,
        **fs_kwargs,
    ):
        """
        Initialize the raster cache.

        Args:
            cache (Optional[Cache]): A cache instance. Defaults to a temporary FilesystemCache.
        """
        self.document = document

        if cache_url == "memory":
            self.cache = InMemoryCache()
        else:
            self.cache = FilesystemCache(
                cache_url=cache_url or f"temp://{self.document.document_hash}",
                **fs_kwargs,
            )

    def cached_pages(self, name: str) -> List[int]:
        """Retrieve the page numbers that are cached."""
        lookup_key = f"{name}/" if not name.endswith("/") else name

        return [
            int(key.split("/")[-1])
            for key in self.cache.list_prefix(lookup_key)
            if key.count("/") == 1
        ]

    def uncached_pages(self, name: str) -> List[int]:
        """Retrieve the page numbers that are not cached."""
        return [
            page_number
            for page_number in self.document
            if page_number not in self.cached_pages(name)
        ]

    def cache_proportion(self, name: str) -> float:
        """Calculate the proportion of the document that is cached."""
        lookup_key = f"{name}/" if not name.endswith("/") else name

        return min(len(self.cache.list_prefix(lookup_key)) / len(self.document), 1.0)

    def fully_cached(self, name: str) -> bool:
        """Check if the entire document is cached."""
        return self.cache_proportion(name) == 1.0

    def _get_key_for_page(
        self,
        key: str,
        page_number: int,
    ) -> str:
        """Generate a cache key for a given page."""
        return f"{key}/{page_number}"

    def get_images_for_key(
        self,
        key: str,
    ) -> Dict[int, bytes]:
        """
        Retrieve all cached raster images for a given key.

        Args:
            key (str): The cache key.

        Returns:
            Dict[int, bytes]: The raster images for each page.
        """
        images = {}
        for page_number in range(1, len(self.document) + 1):
            image = self.get_image_for_page(key, page_number)
            if image is not None:
                images[page_number] = image
        return images

    def set_images_for_key(
        self,
        key: str,
        images: Dict[int, bytes],
    ):
        """
        Cache raster images for a given key.

        Args:
            key (str): The cache key.
            images (Dict[int, bytes]): The raster images for each page.
        """
        for page_number, image in images.items():
            self.set_image_for_page(key, page_number, image)

    def get_image_for_page(
        self,
        key: str,
        page_number: int,
    ):
        """
        Retrieve a cached raster image for a given page.

        Args:
            key (str): The cache key.
            page_number (int): The page number.

        Returns:
            Optional[bytes]: The raster image bytes.
        """
        cache_key = self._get_key_for_page(key, page_number)

        return self.cache.get(cache_key)

    def set_image_for_page(
        self,
        key: str,
        page_number: int,
        image_bytes: bytes,
    ):
        """
        Cache a raster image for a given page.

        Args:
            key (str): The cache key.
            page_number (int): The page number.
            image_bytes (bytes): The raster image bytes.
        """
        cache_key = self._get_key_for_page(key, page_number)

        self.cache.set(cache_key, image_bytes)

    def clear_image_for_page(
        self,
        key: str,
        page_number: int,
    ):
        """
        Clear the cache for a given page.

        Args:
            key (str): The cache key.
            page_number (int): The page number.
        """
        cache_key = self._get_key_for_page(key, page_number)

        self.cache.pop(cache_key)


class PageRasterizer:
    def __init__(
        self,
        owner: "PageNode" = None,
    ):
        """
        Initialize the PageRasterizer with a cache.

        Args:
            owner (PageNode): The owning PageNode instance.
        """
        self.owner = owner

    @property
    def document_cache(self) -> DocumentRasterCache:
        return self.owner.document.rasterizer.cache

    def rasterize(
        self,
        name: str = DEFAULT_CACHE_KEY,
        *,
        return_mode: Literal["bytes", "pil"] = "bytes",
        dpi: int = 100,
        downscale_size: Optional[Tuple[int, int]] = None,
        resize_mode: ResizeModes = "thumbnail",
        resize_aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
        do_convert: bool = False,
        image_convert_mode: str = "L",
        do_quantize: bool = False,
        quantize_color_count: int = 8,
        max_file_size_bytes: Optional[int] = None,
    ) -> Union[bytes, Image.Image]:
        """
        Rasterize the page with the given parameters, utilizing the cache.

        Args:
            name (Optional[str]): Optional name for the cache key.
            return_mode (Literal["bytes", "pil"]): The format to return the rasterized image.
            dpi (int): Dots per inch for rasterization.
            downscale_size (Optional[Tuple[int, int]]): Downscale size.
            resize_mode (ResizeModes): Resize mode.
            resize_aspect_ratios (Optional[Iterable[AspectRatioRule]]): Aspect ratios to maintain.
            do_convert (bool): Whether to convert the image mode.
            image_convert_mode (str): The mode to convert the image to.
            do_quantize (bool): Whether to quantize the image colors.
            quantize_color_count (int): Number of colors for quantization.
            max_file_size_bytes (Optional[int]): Maximum file size in bytes.

        Returns:
            Union[bytes, Image.Image]: The rasterized image in the specified format.
        """

        rastered = self.document_cache.get_image_for_page(name, self.owner.page_number)

        if rastered is None:
            rastered = self.owner.document.document.rasterize_page(
                self.owner.page_number,
                dpi=dpi,
                downscale_size=downscale_size,
                resize_mode=resize_mode,
                resize_aspect_ratios=resize_aspect_ratios,
                do_convert=do_convert,
                image_convert_mode=image_convert_mode,
                do_quantize=do_quantize,
                quantize_color_count=quantize_color_count,
                max_file_size_bytes=max_file_size_bytes,
            )
            self.document_cache.set_image_for_page(
                name, self.owner.page_number, rastered
            )

        if return_mode == "pil" and isinstance(rastered, bytes):
            return Image.open(io.BytesIO(rastered)).copy()
        elif return_mode == "bytes" and isinstance(rastered, bytes):
            return rastered
        else:
            raise ValueError("Invalid return_mode specified.")

    def insert_image(
        self,
        name: str,
        page_number: int,
        image_bytes: bytes,
        *,
        skip_validation: bool = False,
    ) -> None:
        """
        Manually insert a rasterized image into the cache.
        """
        if not skip_validation:
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    img.verify()  # This will raise an exception if the image is not valid
            except Exception as e:
                raise ValueError("Invalid image bytes provided.") from e

        # Insert the image into the cache
        self.document_cache.set_image_for_page(name, page_number, image_bytes)

    def rasterize_to_data_uri(
        self,
        name: str = DEFAULT_CACHE_KEY,
        *,
        dpi: int = 100,
        downscale_size: Optional[Tuple[int, int]] = None,
        resize_mode: ResizeModes = "thumbnail",
        resize_aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
        do_convert: bool = False,
        image_convert_mode: str = "L",
        do_quantize: bool = False,
        quantize_color_count: int = 8,
        max_file_size_bytes: Optional[int] = None,
    ) -> str:
        """
        Rasterize the page and return it as a data URI.

        Args:
            name (str): Name for the cache key.
            dpi (int): Dots per inch for rasterization.
            downscale_size (Optional[Tuple[int, int]]): Downscale size.
            resize_mode (ResizeModes): Resize mode.
            resize_aspect_ratios (Optional[Iterable[AspectRatioRule]]): Aspect ratios to maintain.
            do_convert (bool): Whether to convert the image mode.
            image_convert_mode (str): The mode to convert the image to.
            do_quantize (bool): Whether to quantize the image colors.
            quantize_color_count (int): Number of colors for quantization.
            max_file_size_bytes (Optional[int]): Maximum file size in bytes.

        Returns:
            str: The rasterized image as a data URI.
        """
        rastered = self.rasterize(
            name,
            return_mode="bytes",
            dpi=dpi,
            downscale_size=downscale_size,
            resize_mode=resize_mode,
            resize_aspect_ratios=resize_aspect_ratios,
            do_convert=do_convert,
            image_convert_mode=image_convert_mode,
            do_quantize=do_quantize,
            quantize_color_count=quantize_color_count,
            max_file_size_bytes=max_file_size_bytes,
        )

        return f"data:image/png;base64,{base64.b64encode(rastered).decode('utf-8')}"

    def clear_cache(self, name: str = DEFAULT_CACHE_KEY):
        """
        Clear the cache for the given name.
        """
        self.document_cache.clear_image_for_page(name, self.owner.page_number)


class DocumentRasterizer:
    def __init__(
        self,
        owner: "DocumentNode" = None,
        cache_url: Optional[str] = None,
        **fs_kwargs,
    ):
        """
        Initialize the DocumentRasterizer with a cache.

        Args:
            cache (Optional[Cache]): A cache instance. Defaults to a temporary FilesystemCache.
            owner (DocumentNode): The owning DocumentNode instance.
            cache_prefix (str): Prefix for cache keys to avoid collisions.
        """
        self.cache = DocumentRasterCache(
            document=owner.document, cache_url=cache_url, **fs_kwargs
        )
        self.owner = owner

    def rasterize(
        self,
        name: str = DEFAULT_CACHE_KEY,
        *,
        return_mode: Literal["bytes", "pil"] = "bytes",
        dpi: int = 100,
        downscale_size: Optional[Tuple[int, int]] = None,
        resize_mode: ResizeModes = "thumbnail",
        resize_aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
        do_convert: bool = False,
        image_convert_mode: str = "L",
        do_quantize: bool = False,
        quantize_color_count: int = 8,
        max_file_size_bytes: Optional[int] = None,
        render_grayscale: bool = False,
    ) -> List[Union[bytes, Image.Image]]:
        """
        Rasterize the entire document, utilizing the cache.

        Args:
            name (str): Name for the cache key.
            return_mode (Literal["bytes", "pil"]): The format to return the rasterized images.
            dpi (int): Dots per inch for rasterization.
            downscale_size (Optional[Tuple[int, int]]): Downscale size.
            resize_mode (ResizeModes): Resize mode.
            resize_aspect_ratios (Optional[Iterable[AspectRatioRule]]): Aspect ratios to maintain.
            do_convert (bool): Whether to convert the image mode.
            image_convert_mode (str): The mode to convert the image to.
            do_quantize (bool): Whether to quantize the image colors.
            quantize_color_count (int): Number of colors for quantization.
            max_file_size_bytes (Optional[int]): Maximum file size in bytes.
            render_grayscale (bool): Whether to render in grayscale.

        Returns:
            List[Union[bytes, Image.Image]]: The list of rasterized images.
        """
        rasters: List[Union[bytes, Image.Image]] = []

        if self.cache.fully_cached(name):
            rastered_pdf = self.cache.get_images_for_key(name)
        else:
            rastered_pdf = self.owner.document.rasterize_pdf(
                dpi=dpi,
                downscale_size=downscale_size,
                resize_mode=resize_mode,
                resize_aspect_ratios=resize_aspect_ratios,
                do_convert=do_convert,
                image_convert_mode=image_convert_mode,
                do_quantize=do_quantize,
                quantize_color_count=quantize_color_count,
                max_file_size_bytes=max_file_size_bytes,
                render_grayscale=render_grayscale,
            )

            self.cache.set_images_for_key(name, rastered_pdf)

        for page_number, rastered in rastered_pdf.items():
            if return_mode == "pil" and isinstance(rastered, bytes):
                image = Image.open(io.BytesIO(rastered)).copy()
                rasters.append(image)
            elif return_mode == "bytes" and isinstance(rastered, bytes):
                rasters.append(rastered)
            else:
                rasters.append(rastered)

        return rasters
