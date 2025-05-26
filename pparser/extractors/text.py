"""
Text extraction and structure detection
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import pdfplumber

from .base import BaseExtractor
from ..utils import clean_text, chunk_text


class TextExtractor(BaseExtractor):
    """Extract and structure text content from PDF pages"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.heading_patterns = [
            r'^[A-Z\s]{3,}$',  # ALL CAPS headings
            r'^\d+\.?\s+[A-Z]',  # Numbered headings
            r'^[IVX]+\.?\s+[A-Z]',  # Roman numeral headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?$',  # Title case headings
        ]
    
    def extract(self, pdf_path: Path, page_num: int, **kwargs) -> Dict[str, Any]:
        """Extract structured text from a page"""
        
        result = {
            'type': 'text',
            'content': '',
            'structure': [],
            'headings': [],
            'paragraphs': [],
            'lists': [],
            'footnotes': [],
            'raw_text': ''
        }
        
        try:
            # Extract with both libraries for better coverage
            pymupdf_text = self._extract_with_pymupdf(pdf_path, page_num)
            pdfplumber_text = self._extract_with_pdfplumber(pdf_path, page_num)
            
            # Use the better extraction
            if len(pymupdf_text) > len(pdfplumber_text):
                raw_text = pymupdf_text
            else:
                raw_text = pdfplumber_text
            
            result['raw_text'] = raw_text
            
            if not raw_text.strip():
                return result
            
            # Clean and structure the text
            cleaned_text = clean_text(raw_text)
            result['content'] = cleaned_text
            
            # Extract structure elements
            result['headings'] = self._extract_headings(cleaned_text)
            result['paragraphs'] = self._extract_paragraphs(cleaned_text)
            result['lists'] = self._extract_lists(cleaned_text)
            result['footnotes'] = self._extract_footnotes(cleaned_text)
            result['structure'] = self._build_structure(result)
            
        except Exception as e:
            self.logger.error(f"Error extracting text from page {page_num + 1}: {e}")
        
        return result
    
    def _extract_with_pymupdf(self, pdf_path: Path, page_num: int) -> str:
        """Extract text using PyMuPDF"""
        try:
            doc = self._open_with_pymupdf(pdf_path)
            page = doc[page_num]
            
            # Get text with layout information
            text_dict = page.get_text("dict")
            text = self._parse_pymupdf_text_dict(text_dict)
            
            doc.close()
            return text
            
        except Exception as e:
            self.logger.warning(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def _extract_with_pdfplumber(self, pdf_path: Path, page_num: int) -> str:
        """Extract text using pdfplumber"""
        try:
            with self._open_with_pdfplumber(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    return page.extract_text() or ""
            return ""
            
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
            return ""
    
    def _parse_pymupdf_text_dict(self, text_dict: Dict) -> str:
        """Parse PyMuPDF text dictionary to preserve structure"""
        lines = []
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            
            block_lines = []
            for line in block["lines"]:
                line_text = ""
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text.strip():
                        line_text += text
                
                if line_text.strip():
                    block_lines.append(line_text)
            
            if block_lines:
                lines.extend(block_lines)
                lines.append("")  # Add spacing between blocks
        
        return "\n".join(lines)
    
    def _extract_headings(self, text: str) -> List[Dict[str, Any]]:
        """Extract headings from text"""
        headings = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check against heading patterns
            level = self._determine_heading_level(line)
            if level > 0:
                headings.append({
                    'text': line,
                    'level': level,
                    'line_number': i + 1
                })
        
        return headings
    
    def _determine_heading_level(self, line: str) -> int:
        """Determine heading level (0 = not a heading, 1-6 = heading levels)"""
        line = line.strip()
        
        # Check for numbered headings
        if re.match(r'^\d+\.?\s+[A-Z]', line):
            return 1
        elif re.match(r'^\d+\.\d+\.?\s+[A-Z]', line):
            return 2
        elif re.match(r'^\d+\.\d+\.\d+\.?\s+[A-Z]', line):
            return 3
        
        # Check for ALL CAPS (likely main headings)
        if re.match(r'^[A-Z\s]{3,}$', line) and len(line) < 80:
            return 1
        
        # Check for Title Case
        words = line.split()
        if (len(words) >= 2 and len(words) <= 10 and 
            all(word[0].isupper() for word in words if word.isalpha())):
            return 2
        
        return 0
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text"""
        paragraphs = []
        current_paragraph = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            if not line:
                # Empty line - end current paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            elif self._determine_heading_level(line) > 0:
                # Heading - end current paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                # Regular text - add to current paragraph
                current_paragraph.append(line)
        
        # Add final paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return [p for p in paragraphs if len(p.strip()) > 20]  # Filter short paragraphs
    
    def _extract_lists(self, text: str) -> List[Dict[str, Any]]:
        """Extract lists from text"""
        lists = []
        current_list = None
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_list:
                    lists.append(current_list)
                    current_list = None
                continue
            
            # Check for list patterns
            list_item = self._parse_list_item(line)
            if list_item:
                if not current_list:
                    current_list = {
                        'type': list_item['type'],
                        'items': []
                    }
                current_list['items'].append(list_item)
            else:
                if current_list:
                    lists.append(current_list)
                    current_list = None
        
        if current_list:
            lists.append(current_list)
        
        return lists
    
    def _parse_list_item(self, line: str) -> Optional[Dict[str, str]]:
        """Parse a line as a list item"""
        # Bullet points
        if re.match(r'^[•·▪▫‣⁃]\s+', line):
            return {
                'type': 'bullet',
                'text': re.sub(r'^[•·▪▫‣⁃]\s+', '', line),
                'marker': line[0]
            }
        
        # Numbered lists
        if re.match(r'^\d+\.?\s+', line):
            match = re.match(r'^(\d+\.?)\s+(.+)', line)
            if match:
                return {
                    'type': 'numbered',
                    'text': match.group(2),
                    'number': match.group(1)
                }
        
        # Letter lists
        if re.match(r'^[a-zA-Z]\.?\s+', line):
            match = re.match(r'^([a-zA-Z]\.?)\s+(.+)', line)
            if match:
                return {
                    'type': 'lettered',
                    'text': match.group(2),
                    'letter': match.group(1)
                }
        
        # Dash/hyphen lists
        if re.match(r'^[-–—]\s+', line):
            return {
                'type': 'dash',
                'text': re.sub(r'^[-–—]\s+', '', line),
                'marker': line[0]
            }
        
        return None
    
    def _extract_footnotes(self, text: str) -> List[Dict[str, str]]:
        """Extract footnotes from text"""
        footnotes = []
        
        # Look for footnote patterns
        patterns = [
            r'^\d+\.\s+(.+)',  # 1. footnote text
            r'^\*\s+(.+)',     # * footnote text
            r'^†\s+(.+)',      # † footnote text
        ]
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    footnotes.append({
                        'text': match.group(1),
                        'marker': line.split()[0]
                    })
                    break
        
        return footnotes
    
    def _build_structure(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build hierarchical structure from extracted elements"""
        structure = []
        
        # Add headings
        for heading in result['headings']:
            structure.append({
                'type': 'heading',
                'level': heading['level'],
                'content': heading['text'],
                'line': heading['line_number']
            })
        
        # Add paragraphs (simplified)
        for i, paragraph in enumerate(result['paragraphs']):
            structure.append({
                'type': 'paragraph',
                'content': paragraph,
                'index': i
            })
        
        # Add lists
        for i, list_item in enumerate(result['lists']):
            structure.append({
                'type': 'list',
                'list_type': list_item['type'],
                'items': [item['text'] for item in list_item['items']],
                'index': i
            })
        
        return structure
