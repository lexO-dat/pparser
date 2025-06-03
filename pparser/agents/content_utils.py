"""
Specialized content processing utilities for agents.
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProcessingResult:
    """Standardized result container for processing operations."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'data': self.data,
            'metadata': self.metadata,
            'errors': self.errors,
            'warnings': self.warnings,
            'processing_time': self.processing_time
        }


class ContentChunker:
    """Utility for chunking content to manage token limits."""
    
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        # Rough estimate: 1 token ≈ 4 characters
        self.max_chars = max_tokens * 4
    
    def chunk_text(self, text: str, overlap: int = 200) -> List[str]:
        """Split text into manageable chunks with overlap."""
        if len(text) <= self.max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chars
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(end - 200, start)
                sentence_end = None
                
                for pattern in [r'\.\s+', r'\!\s+', r'\?\s+', r'\n\n']:
                    matches = list(re.finditer(pattern, text[search_start:end]))
                    if matches:
                        sentence_end = search_start + matches[-1].end()
                        break
                
                if sentence_end:
                    end = sentence_end
            
            chunk = text[start:end]
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = max(end - overlap, start + 1)
        
        return chunks
    
    def chunk_sections(self, sections: List[Dict[str, Any]], 
                      max_sections_per_chunk: int = 5) -> List[List[Dict[str, Any]]]:
        """Chunk sections based on content size and count."""
        if not sections:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for section in sections:
            # Estimate section size
            section_text = json.dumps(section)
            section_size = len(section_text)
            
            # Check if adding this section exceeds limits
            if (len(current_chunk) >= max_sections_per_chunk or 
                current_size + section_size > self.max_chars):
                
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(section)
            current_size += section_size
        
        # Add remaining sections
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class ContentValidator:
    """Utility for validating processed content."""
    
    @staticmethod
    def validate_markdown(markdown: str) -> ProcessingResult:
        """Validate markdown content and identify issues."""
        errors = []
        warnings = []
        
        # Check for basic structure
        if not markdown.strip():
            errors.append("Markdown content is empty")
            return ProcessingResult(False, markdown, {}, errors, warnings)
        
        # Check for heading structure
        headings = re.findall(r'^#+\s+(.+)$', markdown, re.MULTILINE)
        if not headings:
            warnings.append("No headings found in markdown")
        
        # Check for malformed math delimiters
        malformed_math = re.findall(r'\\\([^)]*\\\)', markdown)
        if malformed_math:
            warnings.append(f"Found {len(malformed_math)} LaTeX-style math delimiters (should use $ instead)")
        
        # Check for malformed tables
        table_lines = [line for line in markdown.split('\n') if '|' in line]
        if table_lines:
            table_issues = []
            for i, line in enumerate(table_lines):
                if line.count('|') < 2:
                    table_issues.append(f"Line {i+1}: Incomplete table row")
            
            if table_issues:
                warnings.extend(table_issues)
        
        # Check for broken links
        broken_links = re.findall(r'\[([^\]]*)\]\(\s*\)', markdown)
        if broken_links:
            warnings.append(f"Found {len(broken_links)} empty links")
        
        # Check for excessive blank lines
        excessive_blanks = re.findall(r'\n\s*\n\s*\n\s*\n', markdown)
        if excessive_blanks:
            warnings.append("Excessive blank lines found")
        
        metadata = {
            'heading_count': len(headings),
            'table_line_count': len(table_lines),
            'character_count': len(markdown),
            'line_count': len(markdown.split('\n'))
        }
        
        return ProcessingResult(
            success=len(errors) == 0,
            data=markdown,
            metadata=metadata,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_json_structure(data: Dict[str, Any], 
                               required_fields: List[str]) -> ProcessingResult:
        """Validate JSON structure has required fields."""
        errors = []
        warnings = []
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not data[field]:
                warnings.append(f"Empty value for field: {field}")
        
        # Check for unexpected fields
        expected_fields = set(required_fields)
        actual_fields = set(data.keys())
        unexpected = actual_fields - expected_fields
        
        if unexpected:
            warnings.append(f"Unexpected fields: {', '.join(unexpected)}")
        
        metadata = {
            'field_count': len(data),
            'required_field_count': len(required_fields),
            'missing_fields': [f for f in required_fields if f not in data],
            'unexpected_fields': list(unexpected)
        }
        
        return ProcessingResult(
            success=len(errors) == 0,
            data=data,
            metadata=metadata,
            errors=errors,
            warnings=warnings
        )


class ContentCleaner:
    """Utility for cleaning and normalizing content."""
    
    @staticmethod
    def clean_extracted_text(text: str) -> str:
        """Clean text extracted from PDFs."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Fix common OCR issues
        text = re.sub(r'\bl\b', 'I', text)  # lowercase l to I
        text = re.sub(r'\b0\b', 'O', text)  # zero to O where appropriate
        
        # Clean up line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_headings(text: str) -> str:
        """Normalize heading formats in text."""
        # Convert various heading formats to markdown
        patterns = [
            (r'^([A-Z][A-Z\s]+)$', r'# \1'),  # ALL CAPS to H1
            (r'^(\d+\.\s*[A-Z][^.]+)$', r'## \1'),  # Numbered sections to H2
            (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):$', r'### \1'),  # Title case with colon to H3
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for pattern, replacement in patterns:
                if re.match(pattern, line.strip()):
                    lines[i] = re.sub(pattern, replacement, line.strip())
                    break
        
        return '\n'.join(lines)
    
    @staticmethod
    def fix_common_formatting_issues(markdown: str) -> str:
        """Fix common formatting issues in markdown."""
        # Fix spacing around headers
        markdown = re.sub(r'^(#+)\s*([^#\s])', r'\1 \2', markdown, flags=re.MULTILINE)
        
        # Fix list formatting
        markdown = re.sub(r'^(\s*)-([^\s])', r'\1- \2', markdown, flags=re.MULTILINE)
        markdown = re.sub(r'^(\s*)\d+\.([^\s])', r'\1\2. \2', markdown, flags=re.MULTILINE)
        
        # Fix emphasis formatting
        markdown = re.sub(r'\*([^\s*][^*]*[^\s*])\*', r'*\1*', markdown)
        markdown = re.sub(r'\*\*([^\s*][^*]*[^\s*])\*\*', r'**\1**', markdown)
        
        # Remove trailing spaces
        lines = [line.rstrip() for line in markdown.split('\n')]
        
        # Remove excessive blank lines but preserve intentional spacing
        cleaned_lines = []
        blank_count = 0
        
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:  # Allow max 2 consecutive blank lines
                    cleaned_lines.append(line)
            else:
                blank_count = 0
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


class AssetManager:
    """Utility for managing document assets (images, tables, formulas)."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path.cwd()
        self.asset_counters = {'images': 0, 'tables': 0, 'formulas': 0}
    
    def generate_asset_id(self, asset_type: str, page_num: int = 0) -> str:
        """Generate unique ID for asset."""
        self.asset_counters[asset_type] = self.asset_counters.get(asset_type, 0) + 1
        return f"{asset_type}_{page_num}_{self.asset_counters[asset_type]}"
    
    def create_asset_reference(self, asset_id: str, asset_type: str, 
                              title: str = None, inline: bool = True) -> str:
        """Create markdown reference for asset."""
        if not title:
            title = f"{asset_type.title()} {asset_id}"
        
        if asset_type == 'image':
            if inline:
                return f"![{title}]({asset_id}.png)"
            else:
                return f"See [Figure: {title}](#{asset_id})"
        
        elif asset_type == 'table':
            if inline:
                return f"[Table: {title}](#{asset_id})"
            else:
                return f"See [Table: {title}](#{asset_id})"
        
        elif asset_type == 'formula':
            return f"${{{asset_id}}}$"  # LaTeX reference
        
        else:
            return f"[{title}](#{asset_id})"
    
    def organize_assets_by_context(self, assets: List[Dict[str, Any]], 
                                  text_structure: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Organize assets by their contextual placement."""
        organized = {
            'inline': [],      # Assets to be placed inline with text
            'end_section': [], # Assets to be placed at end of sections
            'appendix': []     # Assets to be placed in appendix
        }
        
        for asset in assets:
            complexity = asset.get('complexity', 'moderate')
            purpose = asset.get('purpose', 'explanation')
            size_score = asset.get('size_score', 5)  # 1-10 scale
            
            # Decision logic for placement
            if purpose in ['reference', 'appendix'] or complexity == 'complex':
                organized['appendix'].append(asset)
            elif size_score > 7 or complexity == 'high':
                organized['end_section'].append(asset)
            else:
                organized['inline'].append(asset)
        
        return organized
