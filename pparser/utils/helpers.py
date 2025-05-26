"""
Utility functions for PPARSER system
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might break markdown
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()


def generate_hash(content) -> str:
    """Generate MD5 hash for content"""
    if isinstance(content, bytes):
        return hashlib.md5(content).hexdigest()[:8]
    elif isinstance(content, str):
        return hashlib.md5(content.encode()).hexdigest()[:8]
    else:
        return hashlib.md5(str(content).encode()).hexdigest()[:8]


def safe_filename(filename: str) -> str:
    """Convert string to safe filename"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]+', '_', filename)  # + to collapse consecutive invalid chars
    filename = re.sub(r'\s+', '_', filename)
    filename = re.sub(r'_+', '_', filename)  # Collapse multiple underscores
    filename = filename.strip('._')
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename or "unnamed"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file"""
    ensure_dir(file_path.parent)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: Path) -> Any:
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_page_bounds(page_obj) -> Tuple[float, float, float, float]:
    """Extract page boundaries from PDF page object"""
    try:
        if hasattr(page_obj, 'rect'):  # PyMuPDF
            rect = page_obj.rect
            return (rect.x0, rect.y0, rect.x1, rect.y1)
        elif hasattr(page_obj, 'bbox'):  # pdfplumber
            return page_obj.bbox
        else:
            return (0, 0, 612, 792)  # Default letter size
    except Exception:
        return (0, 0, 612, 792)


def is_image_valid(image_path: Path, min_size: Tuple[int, int] = (50, 50)) -> bool:
    """Check if image is valid and meets minimum size requirements"""
    try:
        with Image.open(image_path) as img:
            return img.size[0] >= min_size[0] and img.size[1] >= min_size[1]
    except Exception:
        return False


def detect_formula_patterns(text: str) -> List[str]:
    """Detect potential mathematical formulas in text"""
    patterns = [
        r'[a-zA-Z]\s*[=≈≠<>≤≥]\s*[0-9a-zA-Z\+\-\*/\(\)\^\s]+',
        r'\$[^$]+\$',
        r'\\\([^)]+\\\)',
        r'\\\[[^\]]+\\\]',
        r'∫[^∫]*d[xyz]',
        r'∑[^∑]*',
        r'√[^√]*',
        r'[α-ωΑ-Ω]',
        r'[∂∇∆∞∈∉⊂⊃∪∩]'
    ]
    
    formulas = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        formulas.extend(matches)
    
    return formulas


def detect_table_patterns(text: str) -> bool:
    """Detect if text contains table-like structures"""
    # Look for common table indicators
    table_indicators = [
        r'\|[^|]*\|[^|]*\|',  # Pipe-separated
        r'^\s*\w+\s+\w+\s+\w+\s*$',  # Multiple columns
        r'\t.*\t',  # Tab-separated
        r':\s*\d+[.,]\d+',  # Numbers with colons
    ]
    
    for pattern in table_indicators:
        if re.search(pattern, text, re.MULTILINE):
            return True
    
    return False


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    if not text or chunk_size <= 0:
        return []
    
    # Ensure overlap is less than chunk_size to prevent infinite loop
    overlap = min(overlap, chunk_size - 1)
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
            
        # Move start forward by chunk_size minus overlap
        start = start + chunk_size - overlap
        
    return chunks


def merge_overlapping_boxes(boxes: List[Tuple[float, float, float, float]], 
                           threshold: float = 10.0) -> List[Tuple[float, float, float, float]]:
    """Merge overlapping bounding boxes"""
    if not boxes:
        return []
    
    # Sort by x0 coordinate
    sorted_boxes = sorted(boxes)
    merged = [sorted_boxes[0]]
    
    for current in sorted_boxes[1:]:
        last = merged[-1]
        
        # Check if boxes overlap or are very close
        if (current[0] <= last[2] + threshold and 
            current[1] <= last[3] + threshold and
            current[3] >= last[1] - threshold):
            # Merge boxes
            merged[-1] = (
                min(last[0], current[0]),
                min(last[1], current[1]),
                max(last[2], current[2]),
                max(last[3], current[3])
            )
        else:
            merged.append(current)
    
    return merged


def merge_bboxes(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Merge multiple bounding boxes into one"""
    if not bboxes:
        return (0, 0, 0, 0)
    
    x1_min = min(bbox[0] for bbox in bboxes)
    y1_min = min(bbox[1] for bbox in bboxes)
    x2_max = max(bbox[2] for bbox in bboxes)
    y2_max = max(bbox[3] for bbox in bboxes)
    
    return (x1_min, y1_min, x2_max, y2_max)


def calculate_overlap(bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
    """Calculate overlap percentage between two bounding boxes"""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # Calculate intersection
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    
    if area1 == 0 or area2 == 0:
        return 0.0
    
    return intersection / min(area1, area2)


def validate_bbox(bbox: Tuple[float, float, float, float]) -> bool:
    """Validate bounding box coordinates"""
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    return x1 < x2 and y1 < y2 and all(isinstance(coord, (int, float)) for coord in bbox)


def get_file_hash(file_path: Path) -> str:
    """Get MD5 hash of file content"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"
