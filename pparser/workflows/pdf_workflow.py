"""
PDF processing workflow using LangGraph for multiagent orchestration.

This module implements the main PDF to Markdown conversion workflow using LangGraph
to coordinate various specialized agents and extractors.
"""

# TODO: i have to implement / correct the iterative markdown validation and construction 

from typing import Dict, Any, List, Optional, TypedDict
import asyncio
from pathlib import Path
import json

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage

from ..config import Config
from ..utils.logger import get_logger
from ..extractors import (
    TextExtractor, ImageExtractor, TableExtractor, 
    FormulaExtractor, FormExtractor
)
from ..agents import (
    TextAnalysisAgent, ImageAnalysisAgent, TableAnalysisAgent,
    FormulaAnalysisAgent, FormAnalysisAgent
)

logger = get_logger(__name__)


class WorkflowState(TypedDict):
    """State object for the PDF processing workflow."""
    pdf_path: str
    output_dir: str
    current_page: int
    total_pages: int
    raw_extractions: Dict[str, Any]
    processed_content: Dict[str, Any]
    structure_map: Dict[str, Any]
    final_markdown: str
    errors: List[str]
    status: str
    metadata: Dict[str, Any]


class PDFWorkflow:
    """
    LangGraph-based workflow for PDF to Markdown conversion.
    
    This class orchestrates the multiagent system to process PDF documents
    and convert them to structured Markdown format.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the PDF workflow.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.logger = logger
        
        # Initialize extractors
        self.text_extractor = TextExtractor(self.config)
        self.image_extractor = ImageExtractor(self.config)
        self.table_extractor = TableExtractor(self.config)
        self.formula_extractor = FormulaExtractor(self.config)
        self.form_extractor = FormExtractor(self.config)
        
        # Initialize agents
        self.text_agent = TextAnalysisAgent(self.config)
        self.image_agent = ImageAnalysisAgent(self.config)
        self.table_agent = TableAnalysisAgent(self.config)
        self.formula_agent = FormulaAnalysisAgent(self.config)
        self.form_agent = FormAnalysisAgent(self.config)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for PDF processing."""
        
        # Create workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_processing)
        workflow.add_node("extract_content", self._extract_content)
        workflow.add_node("analyze_text", self._analyze_text)
        workflow.add_node("analyze_images", self._analyze_images)
        workflow.add_node("analyze_tables", self._analyze_tables)
        workflow.add_node("analyze_formulas", self._analyze_formulas)
        workflow.add_node("analyze_forms", self._analyze_forms)
        workflow.add_node("build_structure", self._build_structure)
        workflow.add_node("assemble_markdown", self._assemble_markdown)
        workflow.add_node("validate_output", self._validate_output)
        workflow.add_node("finalize", self._finalize_processing)
        
        # Define the workflow flow
        workflow.set_entry_point("initialize")
        
        workflow.add_edge("initialize", "extract_content")
        workflow.add_edge("extract_content", "analyze_text")
        workflow.add_edge("analyze_text", "analyze_images")
        workflow.add_edge("analyze_images", "analyze_tables")
        workflow.add_edge("analyze_tables", "analyze_formulas")
        workflow.add_edge("analyze_formulas", "analyze_forms")
        workflow.add_edge("analyze_forms", "build_structure")
        workflow.add_edge("build_structure", "assemble_markdown")
        workflow.add_edge("assemble_markdown", "validate_output")
        workflow.add_edge("validate_output", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow

    async def _initialize_processing(self, state: WorkflowState) -> WorkflowState:
        """Initialize the PDF processing workflow."""
        self.logger.info(f"Initializing PDF processing for: {state['pdf_path']}")
        
        try:
            pdf_path = Path(state['pdf_path'])
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Initialize state
            state['status'] = 'initializing'
            state['errors'] = []
            state['raw_extractions'] = {}
            state['processed_content'] = {}
            state['structure_map'] = {}
            state['final_markdown'] = ''
            state['metadata'] = {
                'pdf_file': str(pdf_path.name),
                'processing_started': True,
                'extraction_methods': ['text', 'images', 'tables', 'formulas', 'forms']
            }
            
            self.logger.info("PDF processing initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing PDF processing: {str(e)}")
            state['errors'].append(f"Initialization error: {str(e)}")
            state['status'] = 'error'
        
        return state

    async def _extract_content(self, state: WorkflowState) -> WorkflowState:
        """Extract all content types from the PDF."""
        self.logger.info("Starting content extraction")
        
        try:
            pdf_path = state['pdf_path']
            
            # Extract text content
            self.logger.info("Extracting text content...")
            text_data_list = self.text_extractor.extract_all_pages(pdf_path)
            text_data = self._consolidate_text_data(text_data_list)
            state['raw_extractions']['text'] = text_data
            
            # Extract images
            self.logger.info("Extracting images...")
            image_data_list = self.image_extractor.extract_all_pages(pdf_path)
            image_data = self._consolidate_image_data(image_data_list)
            state['raw_extractions']['images'] = image_data
            
            # Extract tables
            self.logger.info("Extracting tables...")
            table_data_list = self.table_extractor.extract_all_pages(pdf_path)
            table_data = self._consolidate_table_data(table_data_list)
            state['raw_extractions']['tables'] = table_data
            
            # Extract formulas
            self.logger.info("Extracting formulas...")
            formula_data_list = self.formula_extractor.extract_all_pages(pdf_path)
            formula_data = self._consolidate_formula_data(formula_data_list)
            state['raw_extractions']['formulas'] = formula_data
            
            # Extract forms
            self.logger.info("Extracting forms...")
            form_data_list = self.form_extractor.extract_all_pages(pdf_path)
            form_data = self._consolidate_form_data(form_data_list)
            state['raw_extractions']['forms'] = form_data
            
            state['status'] = 'extracted'
            self.logger.info("Content extraction completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during content extraction: {str(e)}")
            state['errors'].append(f"Extraction error: {str(e)}")
            state['status'] = 'error'
        
        return state

    async def _analyze_text(self, state: WorkflowState) -> WorkflowState:
        """Analyze extracted text content using text analysis agent."""
        self.logger.info("Analyzing text content")
        
        try:
            if 'text' in state['raw_extractions']:
                text_data = state['raw_extractions']['text']
                content_length = len(text_data.get('content', ''))
                self.logger.info(f"Text data to analyze: content length = {content_length}")
                
                # Skip expensive analysis for now - use simplified approach
                analyzed_text = {
                    'content': text_data.get('content', ''),
                    'markdown': text_data.get('content', ''),
                    'structure': {'headings': [], 'sections': []},
                    'metadata': {'analysis_type': 'simplified', 'content_length': content_length}
                }
                
                state['processed_content']['text'] = analyzed_text
                self.logger.info("Text analysis completed")
            else:
                self.logger.warning("No text data found for analysis")
                
        except Exception as e:
            self.logger.error(f"Error analyzing text: {str(e)}")
            state['errors'].append(f"Text analysis error: {str(e)}")
        
        return state

    async def _analyze_images(self, state: WorkflowState) -> WorkflowState:
        """Analyze extracted images using image analysis agent."""
        self.logger.info("Analyzing images")
        
        try:
            if 'images' in state['raw_extractions']:
                image_data = state['raw_extractions']['images']
                # Use simplified approach for now
                analyzed_images = {
                    'items': image_data.get('images', []),
                    'metadata': {'analysis_type': 'simplified', 'count': len(image_data.get('images', []))}
                }
                state['processed_content']['images'] = analyzed_images
                self.logger.info("Image analysis completed")
            else:
                self.logger.warning("No image data found for analysis")
                
        except Exception as e:
            self.logger.error(f"Error analyzing images: {str(e)}")
            state['errors'].append(f"Image analysis error: {str(e)}")
        
        return state

    async def _analyze_tables(self, state: WorkflowState) -> WorkflowState:
        """Analyze extracted tables using table analysis agent."""
        self.logger.info("Analyzing tables")
        
        try:
            if 'tables' in state['raw_extractions']:
                table_data = state['raw_extractions']['tables']
                # Use simplified approach for now
                analyzed_tables = {
                    'items': table_data.get('tables', []),
                    'metadata': {'analysis_type': 'simplified', 'count': len(table_data.get('tables', []))}
                }
                state['processed_content']['tables'] = analyzed_tables
                self.logger.info("Table analysis completed")
            else:
                self.logger.warning("No table data found for analysis")
                
        except Exception as e:
            self.logger.error(f"Error analyzing tables: {str(e)}")
            state['errors'].append(f"Table analysis error: {str(e)}")
        
        return state

    async def _analyze_formulas(self, state: WorkflowState) -> WorkflowState:
        """Analyze extracted formulas using formula analysis agent."""
        self.logger.info("Analyzing formulas")
        
        try:
            if 'formulas' in state['raw_extractions']:
                formula_data = state['raw_extractions']['formulas']
                
                # Process formulas in batches to reduce token usage
                batch_size = 10
                formulas = formula_data.get('formulas', [])
                processed_formulas = []
                
                for i in range(0, len(formulas), batch_size):
                    batch = formulas[i:i + batch_size]
                    # Process batch
                    for formula in batch:
                        # Ensure formula has proper LaTeX formatting
                        if 'latex' not in formula and 'content' in formula:
                            formula['latex'] = formula['content']
                        processed_formulas.append(formula)
                
                analyzed_formulas = {
                    'items': processed_formulas,
                    'metadata': {
                        'analysis_type': 'batch_processed',
                        'count': len(processed_formulas),
                        'batches': (len(formulas) + batch_size - 1) // batch_size
                    }
                }
                state['processed_content']['formulas'] = analyzed_formulas
                self.logger.info(f"Formula analysis completed: {len(processed_formulas)} formulas processed")
            else:
                self.logger.warning("No formula data found for analysis")
                
        except Exception as e:
            self.logger.error(f"Error analyzing formulas: {str(e)}")
            state['errors'].append(f"Formula analysis error: {str(e)}")
        
        return state

    async def _analyze_forms(self, state: WorkflowState) -> WorkflowState:
        """Analyze extracted forms using form analysis agent."""
        self.logger.info("Analyzing forms")
        
        try:
            if 'forms' in state['raw_extractions']:
                form_data = state['raw_extractions']['forms']
                # Use simplified approach for now
                analyzed_forms = {
                    'items': form_data.get('forms', []),
                    'metadata': {'analysis_type': 'simplified', 'count': len(form_data.get('forms', []))}
                }
                state['processed_content']['forms'] = analyzed_forms
                self.logger.info("Form analysis completed")
            else:
                self.logger.warning("No form data found for analysis")
                
        except Exception as e:
            self.logger.error(f"Error analyzing forms: {str(e)}")
            state['errors'].append(f"Form analysis error: {str(e)}")
        
        return state

    async def _build_structure(self, state: WorkflowState) -> WorkflowState:
        """Build the document structure map from all processed content."""
        self.logger.info("Building document structure")
        
        try:
            # Create structure map from processed content
            structure_map = {
                'document_metadata': state['metadata'],
                'content_sections': [],
                'assets': {
                    'images': [],
                    'tables': [],
                    'formulas': [],
                    'forms': []
                }
            }
            
            # Process text structure to create sections
            if 'text' in state['processed_content']:
                text_content = state['processed_content']['text']
                if 'structure' in text_content:
                    structure_map['content_sections'] = text_content['structure']
            
            # Add assets to structure map
            for content_type in ['images', 'tables', 'formulas', 'forms']:
                if content_type in state['processed_content']:
                    content_data = state['processed_content'][content_type]
                    if 'items' in content_data:
                        structure_map['assets'][content_type] = content_data['items']
            
            state['structure_map'] = structure_map
            self.logger.info("Document structure built successfully")
            
        except Exception as e:
            self.logger.error(f"Error building structure: {str(e)}")
            state['errors'].append(f"Structure building error: {str(e)}")
        
        return state

    async def _assemble_markdown(self, state: WorkflowState) -> WorkflowState:
        """Assemble the final Markdown document from all processed content."""
        self.logger.info("Assembling final Markdown")
        
        try:
            # Create structured markdown document
            markdown_parts = []
            
            # Add document header
            pdf_name = Path(state['pdf_path']).stem
            markdown_parts.append(f"# {pdf_name}")
            markdown_parts.append(f"*Converted from PDF using PPARSER*")
            markdown_parts.append("")  # Empty line
            
            # Process content in a single pass to avoid duplicates
            content_sections = []
            asset_sections = []
            toc_items = []
            
            # Process text content first
            if 'text' in state['processed_content']:
                text_content = state['processed_content']['text']
                if text_content and ('markdown' in text_content or 'content' in text_content):
                    content_text = text_content.get('markdown', text_content.get('content', ''))
                    if content_text:
                        text_sections = self._create_text_sections(content_text)
                        content_sections.extend(text_sections)
                        for section in text_sections:
                            toc_items.append(f"- [{section['title']}](#{section['anchor']})")
            
            # Process assets in a single pass
            asset_types = {
                'images': ('Images', self._create_images_section),
                'tables': ('Tables', self._create_tables_section),
                'formulas': ('Formulas', self._create_formulas_section),
                'forms': ('Forms', self._create_forms_section)
            }
            
            for asset_type, (title, creator) in asset_types.items():
                if asset_type in state['processed_content']:
                    content = state['processed_content'][asset_type]
                    if 'items' in content and content['items']:
                        section = creator(content['items'])
                        asset_sections.append(section)
                        toc_items.append(f"- [{title}](#{title.lower()})")
            
            # Add Table of Contents if we have sections
            if toc_items:
                markdown_parts.append("## Table of Contents")
                markdown_parts.extend(toc_items)
                markdown_parts.append("")  # Empty line
            
            # Add main content sections
            for section in content_sections:
                markdown_parts.append(f"## {section['title']}")
                markdown_parts.append(section['content'])
                markdown_parts.append("")  # Empty line
            
            # Add asset sections
            for section in asset_sections:
                markdown_parts.append(section)
                markdown_parts.append("")  # Empty line
            
            state['final_markdown'] = '\n'.join(markdown_parts)
            self.logger.info(f"Structured Markdown assembly completed. Final length: {len(state['final_markdown'])}")
            
        except Exception as e:
            self.logger.error(f"Error assembling Markdown: {str(e)}")
            state['errors'].append(f"Markdown assembly error: {str(e)}")
            
            # Fallback to simple assembly
            try:
                markdown_parts = [f"# {Path(state['pdf_path']).stem}", ""]
                
                if 'text' in state['processed_content']:
                    text_content = state['processed_content']['text']
                    if text_content and 'content' in text_content:
                        markdown_parts.append(text_content['content'])
                
                state['final_markdown'] = '\n'.join(markdown_parts)
                self.logger.info("Fallback Markdown assembly completed")
                
            except Exception as fallback_e:
                self.logger.error(f"Fallback assembly also failed: {str(fallback_e)}")
                state['final_markdown'] = f"# {Path(state['pdf_path']).stem}\n\nError generating content."
        
        return state

    async def _validate_output(self, state: WorkflowState) -> WorkflowState:
        """Validate the final Markdown output."""
        self.logger.info("Validating output")
        
        try:
            # Basic validation checks
            if not state['final_markdown']:
                state['errors'].append("No Markdown content generated")
            elif len(state['final_markdown']) < 10:
                state['errors'].append("Generated Markdown content is too short")
            
            # Check for minimum content requirements
            if 'processed_content' not in state or not state['processed_content']:
                state['errors'].append("No processed content found")
            
            if not state['errors']:
                state['status'] = 'validated'
                self.logger.info("Output validation passed")
            else:
                state['status'] = 'validation_failed'
                self.logger.warning("Output validation failed")
                
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            state['errors'].append(f"Validation error: {str(e)}")
        
        return state

    async def _finalize_processing(self, state: WorkflowState) -> WorkflowState:
        """Finalize the processing and save outputs."""
        self.logger.info("Finalizing processing")
        
        try:
            output_dir = Path(state['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save final Markdown
            markdown_file = output_dir / f"{Path(state['pdf_path']).stem}.md"
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(state['final_markdown'])
            
            # Save processing metadata
            metadata_file = output_dir / f"{Path(state['pdf_path']).stem}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': state['metadata'],
                    'structure_map': state['structure_map'],
                    'processing_errors': state['errors']
                }, f, indent=2)
            
            state['status'] = 'completed'
            state['metadata']['output_files'] = {
                'markdown': str(markdown_file),
                'metadata': str(metadata_file)
            }
            
            self.logger.info(f"Processing completed. Output saved to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error finalizing processing: {str(e)}")
            state['errors'].append(f"Finalization error: {str(e)}")
            state['status'] = 'error'
        
        return state

    async def process_pdf(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a PDF file through the complete workflow.
        
        Args:
            pdf_path: Path to the PDF file to process
            output_dir: Directory to save output files
            
        Returns:
            Dictionary containing processing results and metadata
        """
        self.logger.info(f"Starting PDF processing workflow for: {pdf_path}")
        
        # Initialize workflow state
        initial_state = WorkflowState(
            pdf_path=pdf_path,
            output_dir=output_dir,
            current_page=0,
            total_pages=0,
            raw_extractions={},
            processed_content={},
            structure_map={},
            final_markdown="",
            errors=[],
            status="pending",
            metadata={}
        )
        
        try:
            # Compile and run the workflow
            app = self.workflow.compile()
            final_state = await app.ainvoke(initial_state)
            
            return {
                'status': final_state['status'],
                'output_files': final_state['metadata'].get('output_files', {}),
                'errors': final_state['errors'],
                'metadata': final_state['metadata']
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return {
                'status': 'error',
                'output_files': {},
                'errors': [f"Workflow error: {str(e)}"],
                'metadata': {}
            }

    def _consolidate_text_data(self, text_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate text extraction results from all pages into a single structure"""
        if not text_data_list:
            self.logger.warning("No text data to consolidate")
            return {'type': 'text', 'content': '', 'structure': [], 'headings': [], 'paragraphs': [], 'lists': [], 'footnotes': [], 'raw_text': ''}
        
        self.logger.info(f"Consolidating text data from {len(text_data_list)} pages")
        
        consolidated = {
            'type': 'text',
            'content': '',
            'structure': [],
            'headings': [],
            'paragraphs': [],
            'lists': [],
            'footnotes': [],
            'raw_text': '',
            'pages': text_data_list
        }
        
        for i, page_data in enumerate(text_data_list):
            page_content = page_data.get('content', '')
            page_raw = page_data.get('raw_text', '')
            self.logger.debug(f"Page {i}: content length = {len(page_content)}, raw length = {len(page_raw)}")
            
            if page_data.get('content'):
                consolidated['content'] += page_data['content'] + '\n\n'
            if page_data.get('raw_text'):
                consolidated['raw_text'] += page_data['raw_text'] + '\n\n'
            if page_data.get('headings'):
                consolidated['headings'].extend(page_data['headings'])
            if page_data.get('paragraphs'):
                consolidated['paragraphs'].extend(page_data['paragraphs'])
            if page_data.get('lists'):
                consolidated['lists'].extend(page_data['lists'])
            if page_data.get('footnotes'):
                consolidated['footnotes'].extend(page_data['footnotes'])
        
        return consolidated
    
    def _consolidate_image_data(self, image_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate image extraction results from all pages"""
        if not image_data_list:
            return {'type': 'images', 'images': [], 'metadata': {}}
        
        consolidated = {
            'type': 'images',
            'images': [],
            'metadata': {},
            'pages': image_data_list
        }
        
        for page_data in image_data_list:
            if page_data.get('images'):
                consolidated['images'].extend(page_data['images'])
        
        return consolidated
    
    def _consolidate_table_data(self, table_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate table extraction results from all pages"""
        if not table_data_list:
            return {'type': 'tables', 'tables': [], 'metadata': {}}
        
        consolidated = {
            'type': 'tables',
            'tables': [],
            'metadata': {},
            'pages': table_data_list
        }
        
        for page_data in table_data_list:
            if page_data.get('tables'):
                consolidated['tables'].extend(page_data['tables'])
        
        return consolidated
    
    def _consolidate_formula_data(self, formula_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate formula extraction results from all pages"""
        if not formula_data_list:
            return {'type': 'formulas', 'formulas': [], 'metadata': {}}
        
        consolidated = {
            'type': 'formulas',
            'formulas': [],
            'metadata': {},
            'pages': formula_data_list
        }
        
        for page_data in formula_data_list:
            if page_data.get('formulas'):
                consolidated['formulas'].extend(page_data['formulas'])
        
        return consolidated
    
    def _consolidate_form_data(self, form_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate form extraction results from all pages"""
        if not form_data_list:
            return {'type': 'forms', 'forms': [], 'metadata': {}}
        
        consolidated = {
            'type': 'forms',
            'forms': [],
            'metadata': {},
            'pages': form_data_list
        }
        
        for page_data in form_data_list:
            if page_data.get('forms'):
                consolidated['forms'].extend(page_data['forms'])
        
        return consolidated

    def get_workflow_graph(self) -> str:
        """Get a visual representation of the workflow graph."""
        try:
            app = self.workflow.compile()
            return app.get_graph().draw_mermaid()
        except Exception as e:
            self.logger.error(f"Error generating workflow graph: {str(e)}")
            return "Graph generation failed"
    
    def _create_text_sections(self, text_content: str) -> List[Dict[str, str]]:
        """Create sections from text content by identifying headings and paragraphs."""
        sections = []
        
        # Split content by potential section breaks
        lines = text_content.split('\n')
        current_section = {'title': 'Content', 'anchor': 'content', 'content': []}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for potential headings (lines that are short and might be titles)
            if len(line) < 80 and len(line) > 5 and not line.endswith('.'):
                # Check if this looks like a heading
                words = line.split()
                if len(words) <= 8 and any(word[0].isupper() for word in words):
                    # Save previous section if it has content
                    if current_section['content']:
                        current_section['content'] = '\n'.join(current_section['content'])
                        sections.append(current_section)
                    
                    # Start new section
                    anchor = line.lower().replace(' ', '-').replace('/', '-')
                    anchor = ''.join(c for c in anchor if c.isalnum() or c == '-')
                    current_section = {
                        'title': line,
                        'anchor': anchor,
                        'content': []
                    }
                    continue
            
            # Add line to current section
            current_section['content'].append(line)
        
        # Add final section
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)
        
        return sections
    
    def _create_images_section(self, images: List[Dict]) -> str:
        """Create a Markdown section for images."""
        lines = ["## Images"]
        
        for i, image in enumerate(images, 1):
            lines.append(f"### Image {i}")
            
            if 'description' in image:
                lines.append(f"**Description:** {image['description']}")
            
            if 'filename' in image:
                lines.append(f"![Image {i}]({image['filename']})")
            
            if 'width' in image and 'height' in image:
                lines.append(f"**Dimensions:** {image['width']}x{image['height']}")
            
            lines.append("")  # Empty line between images
        
        return '\n'.join(lines)
    
    def _create_tables_section(self, tables: List[Dict]) -> str:
        """Create a Markdown section for tables."""
        lines = ["## Tables"]
        
        for i, table in enumerate(tables, 1):
            lines.append(f"### Table {i}")
            
            if 'markdown' in table:
                lines.append(table['markdown'])
            elif 'content' in table:
                # Convert basic content to markdown table if possible
                lines.append(str(table['content']))
            
            lines.append("")  # Empty line between tables
        
        return '\n'.join(lines)
    
    def _create_formulas_section(self, formulas: List[Dict]) -> str:
        """Create a Markdown section for formulas."""
        lines = ["## Mathematical Formulas"]
        
        for i, formula in enumerate(formulas, 1):
            lines.append(f"### Formula {i}")
            
            if 'latex' in formula:
                # Use proper math delimiters for inline and block formulas
                if formula.get('type') == 'inline':
                    lines.append(f"${formula['latex']}$")
                else:
                    lines.append(f"$$\n{formula['latex']}\n$$")
                
                # Add explanation if available
                if formula.get('explanation'):
                    lines.append(f"\n*{formula['explanation']}*")
            elif 'content' in formula:
                # Fallback for non-LaTeX formulas
                lines.append(f"```math\n{str(formula['content'])}\n```")
            
            lines.append("")  # Empty line between formulas
        
        return '\n'.join(lines)
    
    def _create_forms_section(self, forms: List[Dict]) -> str:
        """Create a Markdown section for forms."""
        lines = ["## Forms and Fields"]
        
        for i, form in enumerate(forms, 1):
            lines.append(f"### Form {i}")
            
            if 'fields' in form:
                lines.append("**Fields:**")
                for field in form['fields']:
                    field_name = field.get('name', f'Field {len(lines)}')
                    field_type = field.get('type', 'text')
                    lines.append(f"- {field_name} ({field_type})")
            elif 'content' in form:
                lines.append(str(form['content']))
            
            lines.append("")  # Empty line between forms
        
        return '\n'.join(lines)
