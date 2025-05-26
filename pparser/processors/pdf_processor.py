"""
Main PDF processor class that coordinates the entire conversion workflow.

This module provides the primary interface for converting individual PDF files
to structured Markdown using the multiagent system.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import asyncio
import time

from ..config import Config
from ..utils.logger import get_logger
from ..workflows.pdf_workflow import PDFWorkflow
from ..agents import StructureBuilderAgent, QualityValidatorAgent

logger = get_logger(__name__)


class PDFProcessor:
    """
    Main processor for converting PDF files to structured Markdown.
    
    This class provides a high-level interface for the complete PDF to Markdown
    conversion process, integrating all extractors, agents, and workflows.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the PDF processor.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.logger = logger
        
        # Initialize workflow and specialized agents
        self.workflow = PDFWorkflow(self.config)
        self.structure_builder = StructureBuilderAgent(self.config)
        self.quality_validator = QualityValidatorAgent(self.config)

    async def process(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        quality_check: bool = True,
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Process a PDF file and convert it to structured Markdown.
        
        Args:
            pdf_path: Path to the PDF file to process
            output_dir: Directory to save output files. If None, uses same directory as PDF
            quality_check: Whether to perform quality validation and improvement
            return_metadata: Whether to include detailed metadata in the result
            
        Returns:
            Dictionary containing processing results, output files, and metadata
        """
        start_time = time.time()
        
        # Validate inputs
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if output_dir is None:
            output_dir = pdf_path.parent / f"{pdf_path.stem}_output"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting PDF processing: {pdf_path}")
        
        try:
            # Run the main workflow
            workflow_result = await self.workflow.process_pdf(str(pdf_path), str(output_dir))
            
            if workflow_result['status'] == 'error':
                return {
                    'status': 'error',
                    'pdf_file': str(pdf_path),
                    'errors': workflow_result['errors'],
                    'processing_time': time.time() - start_time
                }
            
            # Enhanced processing with structure building and quality validation
            enhanced_result = await self._enhance_output(
                workflow_result,
                str(output_dir),
                quality_check
            )
            
            # Prepare final result
            result = {
                'status': enhanced_result.get('final_status', workflow_result['status']),
                'pdf_file': str(pdf_path),
                'output_directory': str(output_dir),
                'output_files': enhanced_result.get('output_files', workflow_result.get('output_files', {})),
                'processing_time': time.time() - start_time,
                'quality_score': enhanced_result.get('quality_score', 0),
                'errors': workflow_result.get('errors', []) + enhanced_result.get('errors', [])
            }
            
            if return_metadata:
                result['metadata'] = {
                    'workflow_metadata': workflow_result.get('metadata', {}),
                    'enhancement_metadata': enhanced_result.get('metadata', {}),
                    'quality_report': enhanced_result.get('quality_report', {}),
                    'structure_info': enhanced_result.get('structure_info', {})
                }
            
            self.logger.info(f"PDF processing completed: {pdf_path} (Quality: {result['quality_score']:.1f}/100)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return {
                'status': 'error',
                'pdf_file': str(pdf_path),
                'errors': [f"Processing error: {str(e)}"],
                'processing_time': time.time() - start_time
            }

    async def _enhance_output(
        self,
        workflow_result: Dict[str, Any],
        output_dir: str,
        quality_check: bool
    ) -> Dict[str, Any]:
        """
        Enhance the workflow output with structure building and quality validation.
        
        Args:
            workflow_result: Result from the main PDF workflow
            output_dir: Output directory for enhanced files
            quality_check: Whether to perform quality validation
            
        Returns:
            Dictionary containing enhanced processing results
        """
        try:
            enhanced_result = {
                'final_status': workflow_result['status'],
                'output_files': workflow_result.get('output_files', {}),
                'errors': [],
                'quality_score': 0,
                'metadata': {},
                'quality_report': {},
                'structure_info': {}
            }
            
            # Read the initial markdown output
            output_files = workflow_result.get('output_files', {})
            markdown_file = output_files.get('markdown', '')
            
            if not markdown_file or not Path(markdown_file).exists():
                enhanced_result['errors'].append("No markdown file found from workflow")
                return enhanced_result
            
            with open(markdown_file, 'r', encoding='utf-8') as f:
                initial_markdown = f.read()
            
            # Load workflow metadata for content analysis
            metadata_file = output_files.get('metadata', '')
            workflow_metadata = {}
            if metadata_file and Path(metadata_file).exists():
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    workflow_metadata = json.load(f)
            
            # Extract processed content for structure building
            structure_map = workflow_metadata.get('structure_map', {})
            
            # Build enhanced structure if we have content
            if structure_map:
                self.logger.info("Building enhanced document structure")
                structure_result = await self.structure_builder.process({
                    'text': {'markdown': initial_markdown},
                    'images': {'items': structure_map.get('assets', {}).get('images', [])},
                    'tables': {'items': structure_map.get('assets', {}).get('tables', [])},
                    'formulas': {'items': structure_map.get('assets', {}).get('formulas', [])},
                    'forms': {'items': structure_map.get('assets', {}).get('forms', [])}
                })
                
                if structure_result['status'] == 'success':
                    enhanced_markdown = structure_result['markdown']
                    enhanced_result['structure_info'] = structure_result.get('metadata', {})
                    
                    # Save enhanced markdown
                    enhanced_markdown_file = Path(output_dir) / f"{Path(markdown_file).stem}_enhanced.md"
                    with open(enhanced_markdown_file, 'w', encoding='utf-8') as f:
                        f.write(enhanced_markdown)
                    
                    enhanced_result['output_files']['enhanced_markdown'] = str(enhanced_markdown_file)
                    self.logger.info("Enhanced structure built successfully")
                else:
                    enhanced_markdown = initial_markdown
                    enhanced_result['errors'].append(f"Structure building failed: {structure_result.get('error', 'Unknown error')}")
            else:
                enhanced_markdown = initial_markdown
            
            # Perform quality validation if requested
            if quality_check:
                self.logger.info("Performing quality validation")
                
                quality_result = await self.quality_validator.process({
                    'markdown': enhanced_markdown,
                    'original_content': workflow_metadata.get('structure_map', {}),
                    'structure_map': structure_map
                })
                
                if quality_result['status'] == 'success':
                    enhanced_result['quality_score'] = quality_result['quality_score']
                    enhanced_result['quality_report'] = {
                        'validation_checks': quality_result['validation_checks'],
                        'improvements': quality_result['improvements'],
                        'recommendations': quality_result['recommendations']
                    }
                    
                    # Save improved markdown if different
                    improved_markdown = quality_result['improved_markdown']
                    if improved_markdown != enhanced_markdown:
                        improved_markdown_file = Path(output_dir) / f"{Path(markdown_file).stem}_improved.md"
                        with open(improved_markdown_file, 'w', encoding='utf-8') as f:
                            f.write(improved_markdown)
                        
                        enhanced_result['output_files']['improved_markdown'] = str(improved_markdown_file)
                    
                    # Save quality report
                    quality_report_file = Path(output_dir) / f"{Path(markdown_file).stem}_quality_report.json"
                    with open(quality_report_file, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(enhanced_result['quality_report'], f, indent=2)
                    
                    enhanced_result['output_files']['quality_report'] = str(quality_report_file)
                    
                    # Update status based on quality score
                    if enhanced_result['quality_score'] >= 80:
                        enhanced_result['final_status'] = 'high_quality'
                    elif enhanced_result['quality_score'] >= 60:
                        enhanced_result['final_status'] = 'acceptable_quality'
                    else:
                        enhanced_result['final_status'] = 'low_quality'
                    
                    self.logger.info(f"Quality validation completed (Score: {enhanced_result['quality_score']:.1f}/100)")
                else:
                    enhanced_result['errors'].append(f"Quality validation failed: {quality_result.get('error', 'Unknown error')}")
                    enhanced_result['quality_score'] = 50  # Default score for failed validation
            
            enhanced_result['metadata'] = {
                'enhancement_performed': True,
                'structure_building': bool(structure_map),
                'quality_validation': quality_check,
                'final_markdown_length': len(enhanced_markdown),
                'output_files_generated': len(enhanced_result['output_files'])
            }
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Error enhancing output: {str(e)}")
            return {
                'final_status': 'error',
                'output_files': workflow_result.get('output_files', {}),
                'errors': [f"Enhancement error: {str(e)}"],
                'quality_score': 0,
                'metadata': {},
                'quality_report': {},
                'structure_info': {}
            }

    async def process_with_custom_config(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a PDF with custom configuration overrides.
        
        Args:
            pdf_path: Path to the PDF file to process
            output_dir: Directory to save output files
            config_overrides: Dictionary of configuration values to override
            
        Returns:
            Dictionary containing processing results
        """
        # Create temporary config with overrides
        temp_config = Config()
        
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(temp_config, key):
                    setattr(temp_config, key, value)
                else:
                    self.logger.warning(f"Unknown config override: {key}")
        
        # Create temporary processor with custom config
        temp_processor = PDFProcessor(temp_config)
        
        return await temp_processor.process(pdf_path, output_dir)

    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get the current status of the processor.
        
        Returns:
            Dictionary containing processor status information
        """
        return {
            'processor_ready': True,
            'config_loaded': bool(self.config),
            'workflow_initialized': bool(self.workflow),
            'agents_ready': {
                'structure_builder': bool(self.structure_builder),
                'quality_validator': bool(self.quality_validator)
            },
            'supported_features': {
                'text_extraction': True,
                'image_extraction': True,
                'table_extraction': True,
                'formula_extraction': True,
                'form_extraction': True,
                'structure_building': True,
                'quality_validation': True
            }
        }

    def get_workflow_visualization(self) -> str:
        """
        Get a visual representation of the processing workflow.
        
        Returns:
            Mermaid diagram of the workflow
        """
        try:
            return self.workflow.get_workflow_graph()
        except Exception as e:
            self.logger.error(f"Error generating workflow visualization: {str(e)}")
            return "Workflow visualization unavailable"
