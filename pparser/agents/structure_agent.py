"""
Structure building agent for assembling final Markdown documents.

This module contains the StructureBuilder agent that coordinates all processed
content to create a coherent, well-structured Markdown document.
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .base import BaseAgent
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StructureBuilderAgent(BaseAgent):
    """
    Agent specialized in building document structure and generating Markdown.
    
    This agent takes processed content from various extractors and agents,
    analyzes the document structure, and generates well-formatted Markdown.
    """

    def __init__(self, config):
        """
        Initialize the structure builder agent.
        
        Args:
            config: Configuration object containing settings
        """
        super().__init__(
            config=config,
            name="StructureBuilderAgent",
            role="Build document structure and generate Markdown",
            temperature=0.1
        )
        self.llm = ChatOpenAI(
            model=self.config.openai_model,
            temperature=0.1,
            max_tokens=4000,
            openai_api_key=self.config.openai_api_key
        )

    async def process(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the final document structure and assemble Markdown.
        
        Args:
            content_data: Dictionary containing all processed content from extractors and agents
            
        Returns:
            Dictionary containing the structured Markdown document and metadata
        """
        try:
            self.logger.info("Building document structure")
            
            # Extract content components
            text_content = content_data.get('text', {})
            images = content_data.get('images', {}).get('items', [])
            tables = content_data.get('tables', {}).get('items', [])
            formulas = content_data.get('formulas', {}).get('items', [])
            forms = content_data.get('forms', {}).get('items', [])
            
            # Build document outline
            outline = await self._build_document_outline(text_content, images, tables, formulas, forms)
            
            # Assemble sections
            sections = await self._assemble_sections(outline, content_data)
            
            # Generate final Markdown
            final_markdown = await self._generate_final_markdown(sections, outline)
            
            # Create structure metadata
            structure_metadata = {
                'outline': outline,
                'sections': len(sections),
                'content_types': list(content_data.keys()),
                'assets': {
                    'images': len(images),
                    'tables': len(tables),
                    'formulas': len(formulas),
                    'forms': len(forms)
                }
            }
            
            return {
                'status': 'success',
                'markdown': final_markdown,
                'structure': outline,
                'sections': sections,
                'metadata': structure_metadata,
                'content_summary': await self._generate_content_summary(content_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error in structure building: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'markdown': '',
                'structure': {},
                'sections': []
            }

    async def _build_document_outline(
        self, 
        text_content: Dict[str, Any], 
        images: List[Dict], 
        tables: List[Dict], 
        formulas: List[Dict], 
        forms: List[Dict]
    ) -> Dict[str, Any]:
        """Build a logical document outline from all content."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a document structure expert. Create a logical document outline that organizes all content elements into a coherent structure.

                                    Your task:
                                    1. Analyze the text structure and content elements
                                    2. Create a hierarchical outline with sections and subsections
                                    3. Determine where images, tables, formulas, and forms should be placed
                                    4. Ensure logical flow and readability

                                    Return a JSON structure with:
                                    - title: Document title
                                    - sections: Array of section objects with title, content_items, and subsections
                                    - content_placement: Where each asset should be placed

                                    Be comprehensive but maintain readability."""),
            
            HumanMessage(content=f"""
                                Create a document outline from this content:

                                TEXT STRUCTURE:
                                {json.dumps(text_content.get('structure', {}), indent=2)}

                                AVAILABLE ASSETS:
                                - Images: {len(images)} items
                                - Tables: {len(tables)} items  
                                - Formulas: {len(formulas)} items
                                - Forms: {len(forms)} items

                                IMAGE SUMMARIES:
                                {json.dumps([{'id': i.get('id', f'img_{idx}'), 'description': i.get('description', '')[:100]} for idx, i in enumerate(images)], indent=2)}

                                TABLE SUMMARIES:
                                {json.dumps([{'id': t.get('id', f'table_{idx}'), 'description': t.get('description', '')[:100]} for idx, t in enumerate(tables)], indent=2)}

                                FORMULA SUMMARIES:
                                {json.dumps([{'id': f.get('id', f'formula_{idx}'), 'type': f.get('type', '')} for idx, f in enumerate(formulas)], indent=2)}

                                FORM SUMMARIES:
                                {json.dumps([{'id': form.get('id', f'form_{idx}'), 'type': form.get('type', '')} for idx, form in enumerate(forms)], indent=2)}

                                Create a logical document outline in JSON format.
                            """)])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            
            # Parse the JSON response
            outline_text = response.content.strip()
            if outline_text.startswith('```json'):
                outline_text = outline_text[7:-3]
            elif outline_text.startswith('```'):
                outline_text = outline_text[3:-3]
            
            outline = json.loads(outline_text)
            
            # Validate and sanitize the outline
            outline = self._sanitize_outline(outline)
            
            self.logger.info("Document outline created successfully")
            return outline
            
        except Exception as e:
            self.logger.error(f"Error building outline: {str(e)}")
            # Return a basic outline
            return {
                'title': 'Document',
                'sections': [
                    {
                        'title': 'Content',
                        'content_items': ['text'],
                        'subsections': []
                    }
                ],
                'content_placement': {}
            }

    async def _assemble_sections(self, outline: Dict[str, Any], content_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assemble content into structured sections based on the outline."""
        sections = []
        
        try:
            for section_def in outline.get('sections', []):
                section = {
                    'title': section_def.get('title', 'Section'),
                    'content': [],
                    'subsections': []
                }
                
                # Add content items to section
                for content_type in section_def.get('content_items', []):
                    if content_type in content_data:
                        content_item = content_data[content_type]
                        
                        if content_type == 'text':
                            section['content'].append({
                                'type': 'text',
                                'markdown': str(content_item.get('markdown', '')),
                                'metadata': dict(content_item.get('metadata', {}))
                            })
                        else:
                            # Handle asset types
                            items = content_item.get('items', [])
                            for item in items:
                                section['content'].append({
                                    'type': content_type,
                                    'markdown': str(item.get('markdown', '')),
                                    'metadata': dict(item.get('metadata', {})),
                                    'id': str(item.get('id', ''))
                                })
                
                # Process subsections recursively
                for subsection_def in section_def.get('subsections', []):
                    try:
                        subsection = await self._assemble_single_section(subsection_def, content_data)
                        section['subsections'].append(subsection)
                    except Exception as subsection_e:
                        self.logger.warning(f"Error processing subsection: {str(subsection_e)}")
                
                sections.append(section)
                
        except Exception as e:
            self.logger.error(f"Error assembling sections: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
        return sections

    async def _assemble_single_section(self, section_def: Dict[str, Any], content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble a single section."""
        section = {
            'title': section_def.get('title', 'Section'),
            'content': []
        }
        
        try:
            for content_type in section_def.get('content_items', []):
                if content_type in content_data:
                    content_item = content_data[content_type]
                    
                    if content_type == 'text':
                        section['content'].append({
                            'type': 'text',
                            'markdown': str(content_item.get('markdown', '')),
                            'metadata': dict(content_item.get('metadata', {}))
                        })
                    else:
                        items = content_item.get('items', [])
                        for item in items:
                            section['content'].append({
                                'type': content_type,
                                'markdown': str(item.get('markdown', '')),
                                'metadata': dict(item.get('metadata', {})),
                                'id': str(item.get('id', ''))
                            })
        except Exception as e:
            self.logger.error(f"Error assembling single section: {str(e)}")
        
        return section

    async def _chunk_content(self, sections: List[Dict[str, Any]], max_chunk_size: int = 100000) -> List[List[Dict[str, Any]]]:
        """Split content into manageable chunks for processing."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for section in sections:
            section_size = len(json.dumps(section))
            
            if current_size + section_size > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            
            current_chunk.append(section)
            current_size += section_size
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    async def _generate_final_markdown(self, sections: List[Dict[str, Any]], outline: Dict[str, Any]) -> str:
        """Generate the final Markdown document from assembled sections."""
        
        try:
            # Split content into chunks
            content_chunks = await self._chunk_content(sections)
            markdown_parts = []
            
            # Process each chunk
            for i, chunk in enumerate(content_chunks):
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content="""You are a Markdown formatting expert. Create a well-formatted, professional Markdown document from the provided sections.

                                        Requirements:
                                        1. Use proper Markdown hierarchy (# ## ### etc.)
                                        2. Ensure consistent formatting throughout
                                        3. Add appropriate spacing and line breaks
                                        4. Format special elements correctly:
                                           - Code blocks with language specification
                                           - Math equations with LaTeX
                                           - Tables with proper alignment
                                           - Lists (ordered, unordered, definition)
                                           - Index entries with proper indentation
                                           - Footnotes and endnotes
                                        5. Add meaningful section transitions
                                        6. Maintain readability and professional appearance
                                        7. Preserve original document structure
                                        8. Handle special formatting requirements

                                        Output only the final Markdown content, no explanations."""),
                    
                    HumanMessage(content=f"""
                                        Document Outline:
                                        {json.dumps(outline, indent=2)}

                                        Sections to Format (Part {i+1}/{len(content_chunks)}):
                                        {json.dumps(chunk, indent=2)}

                                        Generate the Markdown content for these sections with proper formatting and structure.
                                    """)])
                
                response = await self.llm.ainvoke(prompt.format_messages())
                chunk_markdown = response.content.strip()
                
                # Clean up any code block markers
                if chunk_markdown.startswith('```markdown'):
                    chunk_markdown = chunk_markdown[11:-3]
                elif chunk_markdown.startswith('```'):
                    chunk_markdown = chunk_markdown[3:-3]
                
                markdown_parts.append(chunk_markdown)
            
            # Combine all chunks
            final_markdown = '\n\n'.join(markdown_parts)
            
            # Add table of contents at the beginning if we have multiple sections
            if len(content_chunks) > 1:
                toc = ["## Table of Contents\n"]
                
                # Add main sections
                for section in sections:
                    if 'title' in section:
                        anchor = section['title'].lower().replace(' ', '-')
                        toc.append(f"- [{section['title']}](#{anchor})")
                        
                        # Add subsections if they exist
                        for subsection in section.get('subsections', []):
                            if 'title' in subsection:
                                sub_anchor = subsection['title'].lower().replace(' ', '-')
                                toc.append(f"  - [{subsection['title']}](#{sub_anchor})")
                
                # Add special sections
                special_sections = {
                    'Images': 'images',
                    'Tables': 'tables',
                    'Formulas': 'formulas',
                    'Forms': 'forms',
                    'Index': 'index',
                    'Appendices': 'appendices',
                    'References': 'references'
                }
                
                for title, anchor in special_sections.items():
                    if any(section.get('type') == anchor for section in sections):
                        toc.append(f"- [{title}](#{anchor})")
                
                toc.append("")
                final_markdown = '\n'.join(toc) + final_markdown
            
            self.logger.info("Final Markdown generated successfully")
            return final_markdown
            
        except Exception as e:
            self.logger.error(f"Error generating final Markdown: {str(e)}")
            
            # Fallback: simple concatenation with basic structure
            markdown_parts = [f"# {outline.get('title', 'Document')}\n"]
            
            for section in sections:
                markdown_parts.append(f"\n## {section['title']}\n")
                
                # Handle special formatting for different content types
                for content_item in section['content']:
                    if content_item['markdown']:
                        # Add language specification for code blocks
                        if content_item.get('type') == 'code_block':
                            language = content_item.get('language', '')
                            markdown_parts.append(f"```{language}\n{content_item['markdown']}\n```\n")
                        # Add LaTeX delimiters for math equations
                        elif content_item.get('type') == 'math_equation':
                            markdown_parts.append(f"$$\n{content_item['markdown']}\n$$\n")
                        # Add proper formatting for index entries
                        elif content_item.get('type') == 'index_entry':
                            indent = '  ' * content_item.get('level', 0)
                            markdown_parts.append(f"{indent}- {content_item['markdown']}\n")
                        else:
                            markdown_parts.append(f"{content_item['markdown']}\n")
                
                # Handle subsections
                for subsection in section.get('subsections', []):
                    markdown_parts.append(f"\n### {subsection['title']}\n")
                    for content_item in subsection['content']:
                        if content_item['markdown']:
                            markdown_parts.append(f"{content_item['markdown']}\n")
            
            return '\n'.join(markdown_parts)

    async def _generate_content_summary(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the processed content."""
        summary = {
            'total_content_types': len(content_data),
            'content_breakdown': {}
        }
        
        for content_type, data in content_data.items():
            if isinstance(data, dict):
                if 'items' in data:
                    summary['content_breakdown'][content_type] = {
                        'count': len(data['items']),
                        'has_content': len(data['items']) > 0
                    }
                else:
                    summary['content_breakdown'][content_type] = {
                        'has_content': bool(data.get('markdown') or data.get('content'))
                    }
        
        return summary

    def _sanitize_outline(self, outline: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize the outline to ensure content_items are valid strings."""
        valid_content_types = {'text', 'images', 'tables', 'formulas', 'forms'}
        
        # Ensure outline has required structure
        if not isinstance(outline, dict):
            outline = {}
        
        # Sanitize sections
        sections = outline.get('sections', [])
        if not isinstance(sections, list):
            sections = []
        
        sanitized_sections = []
        for section in sections:
            if not isinstance(section, dict):
                continue
                
            sanitized_section = {
                'title': str(section.get('title', 'Section')),
                'content_items': [],
                'subsections': []
            }
            
            # Sanitize content_items - ensure they are valid strings
            content_items = section.get('content_items', [])
            if not isinstance(content_items, list):
                content_items = []
            
            for item in content_items:
                if isinstance(item, str) and item in valid_content_types:
                    sanitized_section['content_items'].append(item)
                elif isinstance(item, dict) and 'type' in item:
                    # If it's a dict with a type field, extract the type
                    item_type = str(item['type'])
                    if item_type in valid_content_types:
                        sanitized_section['content_items'].append(item_type)
            
            # If no valid content items found, add 'text' as default
            if not sanitized_section['content_items']:
                sanitized_section['content_items'] = ['text']
            
            # Sanitize subsections recursively
            subsections = section.get('subsections', [])
            if isinstance(subsections, list):
                for subsection in subsections:
                    if isinstance(subsection, dict):
                        sanitized_subsection = {
                            'title': str(subsection.get('title', 'Subsection')),
                            'content_items': []
                        }
                        
                        # Sanitize subsection content_items
                        sub_content_items = subsection.get('content_items', [])
                        if isinstance(sub_content_items, list):
                            for item in sub_content_items:
                                if isinstance(item, str) and item in valid_content_types:
                                    sanitized_subsection['content_items'].append(item)
                                elif isinstance(item, dict) and 'type' in item:
                                    item_type = str(item['type'])
                                    if item_type in valid_content_types:
                                        sanitized_subsection['content_items'].append(item_type)
                        
                        if not sanitized_subsection['content_items']:
                            sanitized_subsection['content_items'] = ['text']
                        
                        sanitized_section['subsections'].append(sanitized_subsection)
            
            sanitized_sections.append(sanitized_section)
        
        # Ensure we have at least one section
        if not sanitized_sections:
            sanitized_sections = [{
                'title': 'Content',
                'content_items': ['text'],
                'subsections': []
            }]
        
        return {
            'title': str(outline.get('title', 'Document')),
            'sections': sanitized_sections,
            'content_placement': outline.get('content_placement', {})
        }
