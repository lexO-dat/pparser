"""
Text structure analysis agent
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from .base import BaseAgent
from ..extractors import TextExtractor


class TextAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing and structuring text content"""
    
    def __init__(self, config):
        super().__init__(
            config=config,
            name="TextAnalysisAgent",
            role="Analyze and structure text content from PDF pages",
            temperature=0.1
        )
        self.extractor = TextExtractor(config)
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process PDF page and analyze text structure"""
        
        pdf_path = Path(input_data.get('pdf_path'))
        page_num = input_data.get('page_num', 0)
        
        # Extract raw text content
        extraction_result = self.extractor.extract(pdf_path, page_num)
        
        if not extraction_result.get('content'):
            return {
                'success': False,
                'error': 'No text content found',
                'result': extraction_result
            }
        
        # Enhance structure analysis with LLM
        enhanced_result = self._enhance_structure_analysis(extraction_result)
        
        return {
            'success': True,
            'result': enhanced_result,
            'agent': self.name
        }
    
    def _enhance_structure_analysis(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance structure analysis using LLM"""
        
        content = extraction_result.get('content', '')
        headings = extraction_result.get('headings', [])
        
        system_prompt = f"""You are an expert in document structure analysis. Your role is to analyze text content and improve its hierarchical structure for Markdown conversion.

Tasks:
1. Identify and classify headings by importance level (1-6)
2. Detect document sections and subsections
3. Identify key structural elements:
   - Abstract, introduction, conclusion, references
   - Index/Table of contents
   - Code blocks and technical content
   - Lists (ordered, unordered, definition lists)
   - Tables and figures
   - Appendices
   - Footnotes and endnotes
4. Suggest improvements to heading hierarchy
5. Detect any missing structural elements
6. Identify special formatting needs (code blocks, math equations, etc.)

Return your analysis in JSON format with the following structure:
{{
    "improved_headings": [
        {{"text": "heading text", "level": 1, "type": "title|section|subsection", "line": 1}}
    ],
    "document_sections": [
        {{
            "name": "section name",
            "type": "abstract|introduction|conclusion|references|index|appendix|other",
            "start_line": 1,
            "end_line": 10,
            "special_formatting": ["code_block", "math_equation", "table", "figure"]
        }}
    ],
    "special_elements": {{
        "code_blocks": [
            {{"language": "python|java|etc", "start_line": 1, "end_line": 10}}
        ],
        "index_entries": [
            {{"term": "term", "page": 1, "subentries": []}}
        ],
        "tables": [
            {{"title": "table title", "start_line": 1, "end_line": 5}}
        ],
        "figures": [
            {{"title": "figure title", "start_line": 1, "end_line": 3}}
        ]
    }},
    "structural_improvements": ["suggestion 1", "suggestion 2"],
    "document_type": "academic_paper|book|technical_document|form|other",
    "reading_order": ["section1", "section2", "section3"],
    "formatting_requirements": {{
        "code_blocks": ["language1", "language2"],
        "math_equations": true,
        "special_characters": ["char1", "char2"]
    }}
}}"""
        
        user_content = f"""Analyze this text content:

HEADINGS FOUND:
{chr(10).join([f"Level {h['level']}: {h['text']}" for h in headings])}

FULL TEXT:
{content[:3000]}{"..." if len(content) > 3000 else ""}

Provide your structural analysis."""
        
        messages = self._create_messages(system_prompt, user_content)
        llm_response = self._invoke_llm(messages)
        
        # Parse LLM response and merge with original extraction
        enhanced_structure = self._parse_structure_response(llm_response)
        
        # Merge with original extraction result
        result = extraction_result.copy()
        if enhanced_structure:
            result.update(enhanced_structure)
        
        return result
    
    def _parse_structure_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for structure analysis"""
        
        try:
            import json
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            self.logger.warning(f"Failed to parse structure response: {e}")
        
        return {}


class ContentCleaningAgent(BaseAgent):
    """Agent specialized in cleaning and normalizing text content"""
    
    def __init__(self):
        super().__init__(
            name="ContentCleaningAgent", 
            role="Clean and normalize text content",
            temperature=0.0
        )
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Clean and normalize text content"""
        
        raw_content = input_data.get('content', '')
        
        if not raw_content:
            return {
                'success': False,
                'error': 'No content to clean',
                'result': {'cleaned_content': ''}
            }
        
        # Basic cleaning
        cleaned_content = self._basic_cleaning(raw_content)
        
        # LLM-enhanced cleaning for complex cases
        if self._needs_advanced_cleaning(cleaned_content):
            cleaned_content = self._advanced_cleaning(cleaned_content)
        
        return {
            'success': True,
            'result': {
                'cleaned_content': cleaned_content,
                'original_length': len(raw_content),
                'cleaned_length': len(cleaned_content)
            },
            'agent': self.name
        }
    
    def _basic_cleaning(self, content: str) -> str:
        """Perform basic text cleaning"""
        
        import re
        from ..utils import clean_text
        
        # Use utility function for basic cleaning
        cleaned = clean_text(content)
        
        # Additional cleaning specific to PDF content
        # Remove page numbers and headers/footers patterns
        cleaned = re.sub(r'\n\s*\d+\s*\n', '\n', cleaned)
        cleaned = re.sub(r'\n\s*Page \d+.*?\n', '\n', cleaned, flags=re.IGNORECASE)
        
        # Fix common PDF extraction issues
        cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)  # Missing spaces
        cleaned = re.sub(r'(\w)-\n(\w)', r'\1\2', cleaned)      # Hyphenated words across lines
        
        return cleaned
    
    def _needs_advanced_cleaning(self, content: str) -> bool:
        """Check if content needs advanced LLM cleaning"""
        
        # Check for signs of complex formatting issues
        issues = [
            len(content.split('\n')) / len(content.split()) > 0.5,  # Too many line breaks
            '•' in content and content.count('•') > 10,              # Many bullet points
            content.count('Table') > 3,                             # Multiple tables
            content.count('Figure') > 3,                            # Multiple figures
        ]
        
        return any(issues)
    
    def _advanced_cleaning(self, content: str) -> str:
        """Use LLM for advanced content cleaning"""
        
        system_prompt = """You are an expert text processor. Clean and normalize the following text extracted from a PDF while preserving its structure and meaning.

Tasks:
1. Fix formatting issues (missing spaces, broken words)
2. Normalize whitespace and line breaks
3. Preserve important structure (headings, lists, paragraphs)
4. Remove artifacts like page numbers, headers/footers
5. Ensure text flows naturally

Return only the cleaned text, maintaining the original structure but improving readability."""
        
        user_content = f"Clean this text:\n\n{content[:2000]}"
        
        messages = self._create_messages(system_prompt, user_content)
        cleaned = self._invoke_llm(messages)
        
        return cleaned if cleaned else content
