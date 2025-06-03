"""
Enhanced processor for single PDF processing with advanced features.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..config import Config
from ..utils.logger import get_logger
from .error_handling import ErrorHandler, ErrorSeverity
from .memory_system import MemoryManager

logger = get_logger(__name__)


class EnhancedProcessor:
    """Enhanced processor for single PDF processing with improved capabilities."""
    
    def __init__(self, 
                 agent_factory: 'AgentFactory',
                 quality_check: bool = True,
                 enable_structure_building: bool = True,
                 enable_comprehensive_validation: bool = False,
                 enable_memory: bool = False):
        """Initialize enhanced processor."""
        self.agent_factory = agent_factory
        self.quality_check = quality_check
        self.enable_structure_building = enable_structure_building
        self.enable_comprehensive_validation = enable_comprehensive_validation
        self.enable_memory = enable_memory
        
        self.error_handler = ErrorHandler()
        self.memory_manager = MemoryManager() if enable_memory else None
        
        # Create processing pipeline
        self.pipeline = self._create_processing_pipeline()
        
        logger.info(f"Enhanced processor initialized with {len(self.pipeline)} agents")
    
    def _create_processing_pipeline(self) -> List:
        """Create the processing pipeline based on configuration."""
        pipeline_config = []
        
        # Always include text processing
        pipeline_config.append('text')
        
        # Add content-specific agents
        pipeline_config.extend(['image', 'table', 'formula', 'form'])
        
        # Add structure building if enabled
        if self.enable_structure_building:
            pipeline_config.append('structure')
        
        # Add quality validation if enabled
        if self.quality_check:
            pipeline_config.append('quality')
        
        return self.agent_factory.create_agent_pipeline(pipeline_config)
    
    @ErrorHandler.with_retry(max_attempts=3, delay=2.0)
    async def process_pdf(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a PDF file with enhanced capabilities.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory for results
            
        Returns:
            Processing results with enhanced metadata
        """
        start_time = time.time()
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Starting enhanced PDF processing: {pdf_path}")
            
            # Initialize processing state
            processing_state = {
                'pdf_path': pdf_path,
                'output_dir': str(output_path),
                'start_time': start_time,
                'errors': [],
                'warnings': [],
                'raw_extractions': {},
                'processed_content': {},
                'final_markdown': '',
                'metadata': {},
                'quality_score': 0
            }
            
            # Execute processing pipeline
            for i, agent in enumerate(self.pipeline):
                try:
                    logger.info(f"Executing agent {i+1}/{len(self.pipeline)}: {agent.name}")
                    
                    # Process with agent
                    agent_result = await self._execute_agent(agent, processing_state)
                    
                    # Update state with results
                    self._update_processing_state(processing_state, agent, agent_result)
                    
                    logger.info(f"Agent {agent.name} completed successfully")
                    
                except Exception as e:
                    error_msg = f"Agent {agent.name} failed: {str(e)}"
                    self.error_handler.handle_error(e, ErrorSeverity.MEDIUM, error_msg)
                    processing_state['errors'].append(error_msg)
                    
                    # Continue with other agents unless critical failure
                    if not isinstance(e, (MemoryError, SystemExit)):
                        continue
                    else:
                        raise
            
            # Finalize processing
            result = await self._finalize_processing(processing_state)
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            logger.info(f"Enhanced processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Enhanced processing failed: {str(e)}"
            self.error_handler.handle_error(e, ErrorSeverity.HIGH, error_msg)
            
            return {
                'status': 'error',
                'pdf_file': pdf_path,
                'output_directory': output_dir,
                'errors': [error_msg],
                'processing_time': time.time() - start_time,
                'quality_score': 0
            }
    
    async def _execute_agent(self, agent, processing_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent with proper input preparation."""
        
        # Prepare agent input based on agent type
        agent_input = self._prepare_agent_input(agent, processing_state)
        
        # Record to memory if enabled
        if self.enable_memory and self.memory_manager:
            await self.memory_manager.record_interaction(
                agent.name,
                'processing',
                agent_input,
                {}  # Result will be recorded after execution
            )
        
        # Execute agent
        result = await agent.process_async(agent_input)
        
        # Record result to memory
        if self.enable_memory and self.memory_manager:
            await self.memory_manager.record_interaction(
                agent.name,
                'processing',
                agent_input,
                result
            )
        
        return result
    
    def _prepare_agent_input(self, agent, processing_state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for specific agent types."""
        base_input = {
            'pdf_path': processing_state['pdf_path'],
            'output_dir': processing_state['output_dir']
        }
        
        # Add processed content for later-stage agents
        if hasattr(agent, 'name'):
            if 'structure' in agent.name.lower():
                base_input['processed_content'] = processing_state['processed_content']
            elif 'quality' in agent.name.lower():
                base_input['markdown_content'] = processing_state['final_markdown']
                base_input['original_content'] = processing_state['raw_extractions']
        
        return base_input
    
    def _update_processing_state(self, state: Dict[str, Any], agent, result: Dict[str, Any]):
        """Update processing state with agent results."""
        if not result.get('success', False):
            error_msg = result.get('error', f"Agent {agent.name} failed")
            state['errors'].append(error_msg)
            return
        
        agent_data = result.get('result', {})
        
        # Store results based on agent type
        if hasattr(agent, 'name'):
            agent_name = agent.name.lower()
            
            if 'text' in agent_name:
                state['processed_content']['text'] = agent_data
                if 'markdown' in agent_data:
                    state['final_markdown'] = agent_data['markdown']
            elif 'image' in agent_name:
                state['processed_content']['images'] = agent_data
            elif 'table' in agent_name:
                state['processed_content']['tables'] = agent_data
            elif 'formula' in agent_name:
                state['processed_content']['formulas'] = agent_data
            elif 'form' in agent_name:
                state['processed_content']['forms'] = agent_data
            elif 'structure' in agent_name:
                if 'markdown' in agent_data:
                    state['final_markdown'] = agent_data['markdown']
                state['metadata']['structure_info'] = agent_data.get('metadata', {})
            elif 'quality' in agent_name:
                state['quality_score'] = agent_data.get('quality_score', 0)
                state['metadata']['quality_info'] = agent_data.get('metadata', {})
    
    async def _finalize_processing(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize processing and save outputs."""
        
        # Save final markdown
        output_path = Path(state['output_dir'])
        pdf_name = Path(state['pdf_path']).stem
        
        markdown_file = output_path / f"{pdf_name}.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(state['final_markdown'])
        
        # Save metadata
        import json
        metadata_file = output_path / f"{pdf_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(state['metadata'], f, indent=2)
        
        # Prepare final result
        result = {
            'status': 'completed' if not state['errors'] else 'completed_with_errors',
            'pdf_file': state['pdf_path'],
            'output_directory': state['output_dir'],
            'output_files': {
                'markdown': str(markdown_file),
                'metadata': str(metadata_file)
            },
            'quality_score': state['quality_score'],
            'errors': state['errors'],
            'warnings': state['warnings'],
            'metadata': state['metadata']
        }
        
        # Add enhanced status based on quality score
        if state['quality_score'] >= 80:
            result['status'] = 'high_quality'
        elif state['quality_score'] >= 60:
            result['status'] = 'acceptable_quality'
        elif state['errors']:
            result['status'] = 'low_quality'
        
        # Add structure information
        if state['processed_content']:
            structure_info = {}
            for content_type, content_data in state['processed_content'].items():
                if isinstance(content_data, dict) and 'items' in content_data:
                    structure_info[content_type] = len(content_data['items'])
                elif isinstance(content_data, list):
                    structure_info[content_type] = len(content_data)
            
            result['structure_info'] = structure_info
        
        return result
