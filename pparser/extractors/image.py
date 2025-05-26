"""
Image extraction and processing
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
from PIL import Image

from .base import BaseExtractor
from ..utils import safe_filename, is_image_valid, generate_hash


class ImageExtractor(BaseExtractor):
    """Extract images from PDF pages"""
    
    def __init__(self, config=None, min_size: Tuple[int, int] = (50, 50), output_dir: Optional[Path] = None):
        super().__init__(config)
        self.min_size = min_size
        if output_dir:
            self.output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        elif config and hasattr(config, 'output_dir'):
            self.output_dir = config.output_dir / "images"
        else:
            self.output_dir = Path("images")
        self.output_dir.mkdir(exist_ok=True)
    
    def extract(self, pdf_path: Path, page_num: int, **kwargs) -> Dict[str, Any]:
        """Extract images from a specific page"""
        
        result = {
            'type': 'images',
            'images': [],
            'total_images': 0
        }
        
        try:
            doc = self._open_with_pymupdf(pdf_path)
            page = doc[page_num]
            
            # Get image list from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    image_info = self._extract_image(doc, page, img, page_num, img_index)
                    if image_info:
                        result['images'].append(image_info)
                except Exception as e:
                    self.logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
            
            result['total_images'] = len(result['images'])
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error extracting images from page {page_num + 1}: {e}")
        
        return result
    
    def _extract_image(self, doc, page, img, page_num: int, img_index: int) -> Optional[Dict[str, Any]]:
        """Extract a single image"""
        
        try:
            # Get image reference
            xref = img[0]
            
            # Extract image
            pix = fitz.Pixmap(doc, xref)
            
            # Skip if image is too small
            if pix.width < self.min_size[0] or pix.height < self.min_size[1]:
                pix = None
                return None
            
            # Convert to PIL Image
            if pix.n - pix.alpha < 4:  # GRAY or RGB
                img_data = pix.tobytes("png")
            else:  # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                img_data = pix1.tobytes("png")
                pix1 = None
            
            pix = None
            
            # Open with PIL for validation
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Generate filename
            content_hash = generate_hash(img_data)
            filename = f"page_{page_num + 1}_img_{img_index + 1}_{content_hash}.png"
            filepath = self.output_dir / filename
            
            # Save image
            pil_image.save(filepath, "PNG")
            
            # Get image properties
            image_info = {
                'filename': filename,
                'filepath': str(filepath),
                'width': pil_image.width,
                'height': pil_image.height,
                'format': 'PNG',
                'size_bytes': len(img_data),
                'page_number': page_num + 1,
                'image_index': img_index,
                'description': self._generate_image_description(pil_image),
                'bbox': self._get_image_bbox(page, xref)
            }
            
            return image_info
            
        except Exception as e:
            self.logger.warning(f"Failed to process image: {e}")
            return None
    
    def _generate_image_description(self, image: Image.Image) -> str:
        """Generate a basic description for the image"""
        
        # Analyze image properties
        width, height = image.size
        aspect_ratio = width / height
        
        # Basic classification
        if aspect_ratio > 2:
            shape_desc = "wide"
        elif aspect_ratio < 0.5:
            shape_desc = "tall"
        else:
            shape_desc = "rectangular"
        
        # Size classification
        total_pixels = width * height
        if total_pixels > 500000:
            size_desc = "large"
        elif total_pixels > 100000:
            size_desc = "medium"
        else:
            size_desc = "small"
        
        # Color analysis
        if image.mode == 'L' or image.mode == 'LA':
            color_desc = "grayscale"
        else:
            color_desc = "color"
        
        return f"{size_desc} {shape_desc} {color_desc} image ({width}x{height})"
    
    def _get_image_bbox(self, page, xref: int) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box of image on page"""
        try:
            # Find image in page objects
            for block in page.get_text("dict")["blocks"]:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        # TODO: implement image location, there is no implementation for this, (i didn't know how to do it but i leave it for my future me :D)
                        pass
            
            # Return None if we can't determine position
            return None
            
        except Exception:
            return None
    
    def extract_all_with_positions(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract all images with their positions in the document"""
        all_images = []
        
        try:
            doc = self._open_with_pymupdf(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_images = self.extract(pdf_path, page_num)
                
                # Add position information
                for img in page_images.get('images', []):
                    # Try to get more precise position using page layout
                    rect = self._find_image_rect(page, img)
                    if rect:
                        img['position'] = {
                            'x': rect[0],
                            'y': rect[1],
                            'width': rect[2] - rect[0],
                            'height': rect[3] - rect[1]
                        }
                    
                    all_images.append(img)
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error extracting images with positions: {e}")
        
        return all_images
    
    def _find_image_rect(self, page, image_info: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        """Find the rectangle coordinates of an image on the page"""
        # TODO: this is just a placeholder, so there will be needed to add more analysis of the page layout and the image positioning
        return None
