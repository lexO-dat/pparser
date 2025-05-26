"""
Table extraction and conversion
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber

from .base import BaseExtractor
from ..utils import safe_filename, detect_table_patterns


class TableExtractor(BaseExtractor):
    """Extract and convert tables from PDF pages"""
    
    def __init__(self, config=None, output_dir: Optional[Path] = None):
        super().__init__(config)
        if output_dir:
            self.output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        elif config and hasattr(config, 'output_dir'):
            self.output_dir = config.output_dir / "tables"
        else:
            self.output_dir = Path("tables")
        self.output_dir.mkdir(exist_ok=True)
    
    def extract(self, pdf_path: Path, page_num: int, **kwargs) -> Dict[str, Any]:
        """Extract tables from a specific page"""
        
        result = {
            'type': 'tables',
            'tables': [],
            'total_tables': 0
        }
        
        try:
            # Extract tables using pdfplumber (better for table detection)
            tables = self._extract_with_pdfplumber(pdf_path, page_num)
            
            # Process each table
            for i, table_data in enumerate(tables):
                table_info = self._process_table(table_data, page_num, i)
                if table_info:
                    result['tables'].append(table_info)
            
            result['total_tables'] = len(result['tables'])
            
        except Exception as e:
            self.logger.error(f"Error extracting tables from page {page_num + 1}: {e}")
        
        return result
    
    def _extract_with_pdfplumber(self, pdf_path: Path, page_num: int) -> List[List[List[Optional[str]]]]:
        """Extract tables using pdfplumber"""
        tables = []
        
        try:
            with self._open_with_pdfplumber(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
                    
                    # Also try to find tables in text (fallback)
                    text = page.extract_text()
                    if text and detect_table_patterns(text):
                        text_tables = self._extract_tables_from_text(text)
                        tables.extend(text_tables)
            
        except Exception as e:
            self.logger.warning(f"pdfplumber table extraction failed: {e}")
        
        return tables
    
    def _extract_tables_from_text(self, text: str) -> List[List[List[str]]]:
        """Extract tables from plain text"""
        tables = []
        lines = text.split('\n')
        current_table = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_table and len(current_table) > 1:
                    tables.append(current_table)
                current_table = []
                continue
            
            # Check if line looks like a table row
            if self._is_table_row(line):
                row = self._parse_table_row(line)
                if row:
                    current_table.append(row)
            else:
                if current_table and len(current_table) > 1:
                    tables.append(current_table)
                current_table = []
        
        # Add final table
        if current_table and len(current_table) > 1:
            tables.append(current_table)
        
        return tables
    
    def _is_table_row(self, line: str) -> bool:
        """Check if a line looks like a table row"""
        # Look for multiple columns separated by whitespace or special chars
        separators = ['\t', '|', ':', ';']
        
        for sep in separators:
            if line.count(sep) >= 1:
                return True
        
        # Check for multiple spaced words that might be columns
        words = line.split()
        if len(words) >= 3:
            # Check if there are consistent spacing patterns
            spaces = re.findall(r'\s{2,}', line)
            if len(spaces) >= 1:
                return True
        
        return False
    
    def _parse_table_row(self, line: str) -> Optional[List[str]]:
        """Parse a line into table cells"""
        # Try different separators
        separators = ['\t', '|', '::', '  ']
        
        for sep in separators:
            if sep in line:
                cells = [cell.strip() for cell in line.split(sep)]
                if len(cells) >= 2:
                    return cells
        
        # Fallback: split on multiple spaces
        cells = re.split(r'\s{2,}', line.strip())
        if len(cells) >= 2:
            return cells
        
        return None
    
    def _process_table(self, table_data: List[List[Optional[str]]], page_num: int, table_index: int) -> Optional[Dict[str, Any]]:
        """Process and save a table"""
        
        try:
            # Clean the table data
            cleaned_table = self._clean_table_data(table_data)
            
            if not cleaned_table or len(cleaned_table) < 2:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
            
            # Generate filename
            filename = f"page_{page_num + 1}_table_{table_index + 1}.csv"
            filepath = self.output_dir / filename
            
            # Save as CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            # Generate markdown representation
            markdown_table = self._table_to_markdown(cleaned_table)
            
            table_info = {
                'filename': filename,
                'filepath': str(filepath),
                'page_number': page_num + 1,
                'table_index': table_index,
                'rows': len(cleaned_table),
                'columns': len(cleaned_table[0]) if cleaned_table else 0,
                'markdown': markdown_table,
                'headers': cleaned_table[0] if cleaned_table else [],
                'description': self._generate_table_description(df)
            }
            
            return table_info
            
        except Exception as e:
            self.logger.warning(f"Failed to process table: {e}")
            return None
    
    def _clean_table_data(self, table_data: List[List[Optional[str]]]) -> List[List[str]]:
        """Clean and normalize table data"""
        cleaned = []
        
        for row in table_data:
            if not row:
                continue
            
            # Clean cells
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Clean the cell content
                    cell = str(cell).strip()
                    cell = re.sub(r'\s+', ' ', cell)
                    cleaned_row.append(cell)
            
            # Skip empty rows
            if any(cell.strip() for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        # Ensure all rows have the same number of columns
        if cleaned:
            max_cols = max(len(row) for row in cleaned)
            for row in cleaned:
                while len(row) < max_cols:
                    row.append("")
        
        return cleaned
    
    def _table_to_markdown(self, table_data: List[List[str]]) -> str:
        """Convert table data to Markdown format"""
        if not table_data:
            return ""
        
        lines = []
        
        # Headers
        headers = table_data[0]
        lines.append("| " + " | ".join(headers) + " |")
        
        # Separator
        separator = "|" + "|".join([" --- " for _ in headers]) + "|"
        lines.append(separator)
        
        # Data rows
        for row in table_data[1:]:
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def _generate_table_description(self, df: pd.DataFrame) -> str:
        """Generate a description for the table"""
        
        rows, cols = df.shape
        
        # Analyze data types
        numeric_cols = df.select_dtypes(include=['number']).columns
        text_cols = df.select_dtypes(include=['object']).columns
        
        desc_parts = [f"{rows} rows, {cols} columns"]
        
        if len(numeric_cols) > 0:
            desc_parts.append(f"{len(numeric_cols)} numeric columns")
        
        if len(text_cols) > 0:
            desc_parts.append(f"{len(text_cols)} text columns")
        
        return "Table with " + ", ".join(desc_parts)
    
    def extract_complex_tables(self, pdf_path: Path, page_num: int) -> Dict[str, Any]:
        """Extract complex tables that might span multiple areas"""
        
        result = {
            'type': 'complex_tables',
            'tables': [],
            'total_tables': 0
        }
        
        try:
            # Use multiple extraction strategies
            strategies = [
                self._extract_with_bbox_analysis,
                self._extract_with_line_detection,
                self._extract_with_text_clustering
            ]
            
            all_tables = []
            for strategy in strategies:
                try:
                    tables = strategy(pdf_path, page_num)
                    all_tables.extend(tables)
                except Exception as e:
                    self.logger.warning(f"Strategy failed: {e}")
            
            # Remove duplicates and process
            unique_tables = self._deduplicate_tables(all_tables)
            
            for i, table_data in enumerate(unique_tables):
                table_info = self._process_table(table_data, page_num, i)
                if table_info:
                    result['tables'].append(table_info)
            
            result['total_tables'] = len(result['tables'])
            
        except Exception as e:
            self.logger.error(f"Error extracting complex tables: {e}")
        
        return result
    
    def _extract_with_bbox_analysis(self, pdf_path: Path, page_num: int) -> List[List[List[str]]]:
        """Extract tables by analyzing bounding boxes"""
        # TODO: advanced table detection
        return []
    
    def _extract_with_line_detection(self, pdf_path: Path, page_num: int) -> List[List[List[str]]]:
        """Extract tables by detecting table lines"""
        # TODO: line-based table detection
        return []
    
    def _extract_with_text_clustering(self, pdf_path: Path, page_num: int) -> List[List[List[str]]]:
        """Extract tables by clustering text elements"""
        # TODO: clustering-based table detection
        return []
    
    def _deduplicate_tables(self, tables: List[List[List[str]]]) -> List[List[List[str]]]:
        """Remove duplicate tables"""
        unique = []
        seen = set()
        
        for table in tables:
            # Create a hash of the table content
            table_str = str(table)
            table_hash = hash(table_str)
            
            if table_hash not in seen:
                seen.add(table_hash)
                unique.append(table)
        
        return unique
