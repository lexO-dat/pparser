"""
Base extractor class for PDF content extraction
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import pdfplumber

from ..utils import logger


class BaseExtractor(ABC):
    """Base class for all PDF content extractors"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logger
    
    @abstractmethod
    def extract(self, pdf_path: Path, page_num: int, **kwargs) -> Dict[str, Any]:
        """Extract content from a specific page"""
        pass
    
    def extract_all_pages(self, pdf_path: Path, **kwargs) -> List[Dict[str, Any]]:
        """Extract content from all pages"""
        results = []
        
        try:
            # Use PyMuPDF to get page count
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            for page_num in range(total_pages):
                try:
                    result = self.extract(pdf_path, page_num, **kwargs)
                    if result:
                        result['page_number'] = page_num + 1
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to extract from page {page_num + 1}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to process PDF {pdf_path}: {e}")
        
        return results
    
    def _open_with_pymupdf(self, pdf_path: Path):
        """Open PDF with PyMuPDF"""
        return fitz.open(pdf_path)
    
    def _open_with_pdfplumber(self, pdf_path: Path):
        """Open PDF with pdfplumber"""
        return pdfplumber.open(pdf_path)
