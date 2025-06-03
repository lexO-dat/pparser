"""
Structure building agent for assembling final Markdown documents.

This module contains the StructureBuilder agent that coordinates all processed
content to create a coherent, well-structured Markdown document.
"""

from typing import Dict, Any, List, Optional
import json
import re
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
        """Build a logical document outline from all content using chunked processing to avoid token limits."""
        
        try:
            # First, create a basic structure outline with limited content
            basic_outline = await self._create_basic_outline(text_content)
            
            # Then progressively add assets in manageable chunks
            outline_with_assets = await self._add_assets_to_outline(
                basic_outline, images, tables, formulas, forms
            )
            
            # Validate and sanitize the final outline
            final_outline = self._sanitize_outline(outline_with_assets)
            
            self.logger.info("Document outline created successfully with chunked processing")
            return final_outline
            
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

    async def _create_basic_outline(self, text_content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic document outline from text structure only."""
        
        # Limit text structure to prevent token overflow
        structure = text_content.get('structure', {})
        limited_structure = self._limit_structure_size(structure)
        
        # Build prompt content
        structure_json = json.dumps(limited_structure, indent=2)
        
        prompt_content = f"""
                                Create a basic document outline from this text structure:

                                TEXT STRUCTURE:
                                {structure_json}

                                Create a logical document outline in JSON format.
                            """
        
        # Check token limit before sending
        if not self._check_token_limit(prompt_content, max_tokens=90000):
            self.logger.warning("Text structure still too large, using minimal outline")
            # Use an even more aggressive limit
            limited_structure = self._limit_structure_size(structure, max_items=20)
            structure_json = json.dumps(limited_structure, indent=2)
            prompt_content = f"""
                                Create a basic document outline from this text structure:

                                TEXT STRUCTURE:
                                {structure_json}

                                Create a logical document outline in JSON format.
                            """
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a document structure expert. Create a basic document outline from the text structure.

                                    Your task:
                                    1. Analyze the text structure and create logical sections
                                    2. Create a hierarchical outline with sections and subsections
                                    3. Focus on text organization and flow

                                    Return a JSON structure with:
                                    - title: Document title (extract from content)
                                    - sections: Array of section objects with title and subsections
                                    - content_placement: Empty object (will be populated later)

                                    Keep the outline concise but comprehensive."""),
            
            HumanMessage(content=prompt_content)
        ])
        
        response = await self.llm.ainvoke(prompt.format_messages())
        
        # Parse the JSON response
        outline_text = response.content.strip()
        if outline_text.startswith('```json'):
            outline_text = outline_text[7:-3]
        elif outline_text.startswith('```'):
            outline_text = outline_text[3:-3]
        
        return json.loads(outline_text)

    async def _add_assets_to_outline(
        self, 
        outline: Dict[str, Any], 
        images: List[Dict], 
        tables: List[Dict], 
        formulas: List[Dict], 
        forms: List[Dict]
    ) -> Dict[str, Any]:
        """Add assets to the outline in manageable chunks."""
        
        # Process assets in smaller batches to avoid token limits
        all_assets = []
        
        # Limit asset descriptions to prevent token overflow
        for idx, img in enumerate(images[:20]):  # Limit to 20 images
            all_assets.append({
                'type': 'image',
                'id': img.get('id', f'img_{idx}'),
                'description': img.get('description', '')[:50]  # Truncate descriptions
            })
        
        for idx, table in enumerate(tables[:15]):  # Limit to 15 tables
            all_assets.append({
                'type': 'table',
                'id': table.get('id', f'table_{idx}'),
                'description': table.get('description', '')[:50]
            })
        
        for idx, formula in enumerate(formulas[:25]):  # Limit to 25 formulas
            all_assets.append({
                'type': 'formula',
                'id': formula.get('id', f'formula_{idx}'),
                'formula_type': formula.get('type', '')
            })
        
        for idx, form in enumerate(forms[:10]):  # Limit to 10 forms
            all_assets.append({
                'type': 'form',
                'id': form.get('id', f'form_{idx}'),
                'form_type': form.get('type', '')
            })
        
        # Process assets in chunks of 10
        chunk_size = 10
        final_outline = outline.copy()
        
        for i in range(0, len(all_assets), chunk_size):
            chunk = all_assets[i:i + chunk_size]
            final_outline = await self._integrate_asset_chunk(final_outline, chunk)
        
        return final_outline

    async def _integrate_asset_chunk(self, outline: Dict[str, Any], assets: List[Dict]) -> Dict[str, Any]:
        """Integrate a chunk of assets into the existing outline."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are integrating content assets into a document outline. 
                                    Your task is to place assets (images, tables, formulas, forms) INLINE within the appropriate content sections where they naturally belong, rather than creating separate sections for them.
                                    
                                    Rules:
                                    1. Images should be placed inline within the text where they are referenced or contextually relevant
                                    2. Tables should be embedded within sections where their data is discussed  
                                    3. Formulas should be placed inline within the mathematical content where they appear
                                    4. Forms should be integrated where survey/questionnaire content is mentioned
                                    5. Do NOT create separate "Images", "Tables", "Formulas", or "Forms" sections
                                    6. Instead, add assets to existing content sections using the content_placement field
                                    7. Use section titles like "Content", "Background", "Model Architecture", etc. to place assets contextually
                                    
                                    Update the content_placement object to specify inline placement within existing sections.
                                    The goal is to recreate the original document structure where assets appear integrated with text."""),
            
            HumanMessage(content=f"""
                                Current outline:
                                {json.dumps(outline, indent=2)}

                                Assets to integrate INLINE (not in separate sections):
                                {json.dumps(assets, indent=2)}

                                Return the updated outline with inline placement information for these assets within appropriate existing sections.
                                Focus on contextual placement within "Content", "Background", "Model Architecture" and similar content sections.
                            """)
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            
            # Parse the JSON response
            outline_text = response.content.strip()
            if outline_text.startswith('```json'):
                outline_text = outline_text[7:-3]
            elif outline_text.startswith('```'):
                outline_text = outline_text[3:-3]
            
            updated_outline = json.loads(outline_text)
            return updated_outline
            
        except Exception as e:
            self.logger.warning(f"Error integrating asset chunk: {str(e)}")
            # Return original outline if integration fails
            return outline

    def _limit_structure_size(self, structure: Dict[str, Any], max_items: int = 100) -> Dict[str, Any]:
        """Limit the size of text structure to prevent token overflow."""
        if not isinstance(structure, dict):
            return structure
        
        limited = {}
        item_count = 0
        
        for key, value in structure.items():
            if item_count >= max_items:
                break
                
            if isinstance(value, list):
                # Limit list items
                limited[key] = value[:10]  # Keep first 10 items
                item_count += min(len(value), 10)
            elif isinstance(value, dict):
                # Recursively limit nested dictionaries
                limited[key] = self._limit_structure_size(value, max_items - item_count)
                item_count += len(limited[key])
            elif isinstance(value, str):
                # Truncate long strings
                limited[key] = value[:200] if len(value) > 200 else value
                item_count += 1
            else:
                limited[key] = value
                item_count += 1
        
        return limited

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
                                           - Math equations with $ and $$ delimiters (NOT \( and \) )
                                           - Tables with proper alignment
                                           - Lists (ordered, unordered, definition)
                                           - Index entries with proper indentation
                                           - Footnotes and endnotes
                                        5. Add meaningful section transitions
                                        6. Maintain readability and professional appearance
                                        7. Preserve original document structure
                                        8. Handle special formatting requirements
                                        9. Place images, tables, and formulas inline where they belong in content flow, NOT in separate sections
                                        10. Use ONLY Markdown math delimiters: $ for inline math and $$ for block math

                                        CRITICAL: Mathematical formulas must use $ and $$ delimiters, never \( and \)

                                        Output only the final Markdown content, no explanations."""),
                    
                    HumanMessage(content=f"""
                                        Document Outline:
                                        {json.dumps(outline, indent=2)}

                                        Sections to Format (Part {i+1}/{len(content_chunks)}):
                                        {json.dumps(chunk, indent=2)}

                                        Generate the Markdown content for these sections with proper formatting and structure.
                                        Place all assets (images, tables, formulas) inline where they belong in the content flow.
                                        Use only $ and $$ for mathematical formulas, never \( and \).
                                    """)])
                
                response = await self.llm.ainvoke(prompt.format_messages())
                chunk_markdown = response.content.strip()
                
                # Clean up any code block markers
                if chunk_markdown.startswith('```markdown'):
                    chunk_markdown = chunk_markdown[11:-3]
                elif chunk_markdown.startswith('```'):
                    chunk_markdown = chunk_markdown[3:-3]
                
                # Fix math delimiters - convert LaTeX delimiters to Markdown
                chunk_markdown = self._fix_math_delimiters(chunk_markdown)
                
                markdown_parts.append(chunk_markdown)
            
            # Combine all chunks
            final_markdown = '\n\n'.join(markdown_parts)
            
            # Apply final math delimiter fix to entire document
            final_markdown = self._fix_math_delimiters(final_markdown)
            
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
                        # Add proper math delimiters for math equations
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
            
            fallback_markdown = '\n'.join(markdown_parts)
            # Apply math delimiter fix to fallback content too
            return self._fix_math_delimiters(fallback_markdown)

    def _fix_math_delimiters(self, text: str) -> str:
        """Fix LaTeX math delimiters to use Markdown format."""
        # Convert \( ... \) to $ ... $
        text = re.sub(r'\\\(([^)]*)\\\)', r'$\1$', text)
        
        # Convert \[ ... \] to $$ ... $$
        text = re.sub(r'\\\[([^]]*)\\\]', r'$$\1$$', text)
        
        # Clean up any double delimiters that might have been created
        text = re.sub(r'\$\$\$+', '$$', text)
        text = re.sub(r'\$\$\$([^$]+)\$\$\$', r'$$\1$$', text)
        
        return text

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
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.
        Simple heuristic: ~4 characters per token for English text.
        """
        # Remove extra whitespace and normalize
        normalized_text = re.sub(r'\s+', ' ', text.strip())
        
        # Rough estimation: 4 characters per token
        # This is conservative for GPT models
        estimated_tokens = len(normalized_text) / 4
        
        return int(estimated_tokens)

    def _check_token_limit(self, content: str, max_tokens: int = 100000) -> bool:
        """
        Check if content exceeds token limit.
        Uses conservative limit below GPT-4o-mini's 128k context window.
        """
        estimated = self._estimate_tokens(content)
        if estimated > max_tokens:
            self.logger.warning(f"Content may exceed token limit: {estimated} estimated tokens")
            return False
        return True
