"""
Example usage of the refactored PPARSER agent system.

This example demonstrates how to use the enhanced agent architecture
with improved error handling, memory management, and configuration.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any

from pparser.config import Config
from pparser.agents import (
    AgentFactory, 
    ContentValidator, 
    ContentCleaner,
    MemoryManager
)


class EnhancedPPARSERProcessor:
    """Enhanced PDF processor using refactored agent system."""
    
    def __init__(self, config: Config, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        
        # Initialize enhanced components
        self.agent_factory = AgentFactory(config)
        self.memory_manager = MemoryManager()
        self.content_validator = ContentValidator()
        self.content_cleaner = ContentCleaner()
        
        # Create processing pipeline
        self.pipeline = self.agent_factory.get_default_processing_pipeline(output_dir)
        
    async def process_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Process PDF document using enhanced agent pipeline."""
        
        try:
            # Initialize processing context
            context = {
                'pdf_path': str(pdf_path),
                'output_dir': str(self.output_dir),
                'start_time': None
            }
            
            # Process through pipeline
            results = {}
            
            for agent in self.pipeline:
                agent_name = agent.name
                print(f"Processing with {agent_name}...")
                
                # Record context in agent memory
                agent.add_context('document_processing', context)
                
                # Process based on agent type
                if 'text' in agent_name.lower():
                    result = await self._process_text_content(agent, pdf_path)
                elif 'image' in agent_name.lower():
                    result = await self._process_images(agent, pdf_path)
                elif 'table' in agent_name.lower():
                    result = await self._process_tables(agent, pdf_path)
                elif 'formula' in agent_name.lower():
                    result = await self._process_formulas(agent, pdf_path)
                elif 'form' in agent_name.lower():
                    result = await self._process_forms(agent, pdf_path)
                elif 'structure' in agent_name.lower():
                    result = await self._build_structure(agent, results)
                elif 'quality' in agent_name.lower():
                    result = await self._validate_quality(agent, results)
                else:
                    result = {'success': False, 'error': f'Unknown agent type: {agent_name}'}
                
                # Record result
                if result.get('success'):
                    agent.record_result(f'process_{agent_name}', result)
                    results[agent_name] = result
                else:
                    print(f"Warning: {agent_name} failed: {result.get('error', 'Unknown error')}")
                    results[agent_name] = result
            
            # Generate final output
            final_result = await self._generate_final_output(results)
            
            # Get processing insights
            insights = self._get_processing_insights()
            final_result['processing_insights'] = insights
            
            return final_result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'partial_results': results if 'results' in locals() else {}
            }
    
    async def _process_text_content(self, agent, pdf_path: Path) -> Dict[str, Any]:
        """Process text content with enhanced error handling."""
        try:
            # Example processing - adapt to your actual implementation
            input_data = {'pdf_path': str(pdf_path), 'page_num': 0}
            result = agent.process(input_data)
            
            # Clean and validate result
            if result.get('success') and 'content' in result.get('result', {}):
                content = result['result']['content']
                cleaned_content = self.content_cleaner.clean_extracted_text(content)
                result['result']['cleaned_content'] = cleaned_content
            
            return result
            
        except Exception as e:
            return agent.error_handler.handle_error(e, context={'operation': 'text_processing'})
    
    async def _process_images(self, agent, pdf_path: Path) -> Dict[str, Any]:
        """Process images with enhanced handling."""
        try:
            input_data = {'pdf_path': str(pdf_path), 'page_num': 0}
            result = agent.process(input_data)
            
            # Validate image processing results
            if result.get('success') and 'images' in result.get('result', {}):
                images = result['result']['images']
                
                # Add asset management
                for i, image in enumerate(images):
                    if 'id' not in image:
                        image['id'] = f"img_{i}"
                    if 'placement_hint' not in image:
                        image['placement_hint'] = 'inline'
            
            return result
            
        except Exception as e:
            return agent.error_handler.handle_error(e, context={'operation': 'image_processing'})
    
    async def _process_tables(self, agent, pdf_path: Path) -> Dict[str, Any]:
        """Process tables with enhanced validation."""
        try:
            input_data = {'pdf_path': str(pdf_path), 'page_num': 0}
            result = agent.process(input_data)
            
            # Validate table markdown
            if result.get('success') and 'tables' in result.get('result', {}):
                tables = result['result']['tables']
                
                for table in tables:
                    if 'markdown' in table:
                        validation = self.content_validator.validate_markdown(table['markdown'])
                        table['validation'] = validation.to_dict()
            
            return result
            
        except Exception as e:
            return agent.error_handler.handle_error(e, context={'operation': 'table_processing'})
    
    async def _process_formulas(self, agent, pdf_path: Path) -> Dict[str, Any]:
        """Process formulas with math delimiter validation."""
        try:
            input_data = {'pdf_path': str(pdf_path), 'page_num': 0}
            result = agent.process(input_data)
            
            # Fix math delimiters
            if result.get('success') and 'formulas' in result.get('result', {}):
                formulas = result['result']['formulas']
                
                for formula in formulas:
                    if 'markdown' in formula:
                        # Apply math delimiter fix
                        fixed_markdown = self.content_cleaner.fix_math_delimiters(formula['markdown'])
                        formula['markdown'] = fixed_markdown
            
            return result
            
        except Exception as e:
            return agent.error_handler.handle_error(e, context={'operation': 'formula_processing'})
    
    async def _process_forms(self, agent, pdf_path: Path) -> Dict[str, Any]:
        """Process forms with enhanced structure validation."""
        try:
            input_data = {'pdf_path': str(pdf_path), 'page_num': 0}
            result = agent.process(input_data)
            
            return result
            
        except Exception as e:
            return agent.error_handler.handle_error(e, context={'operation': 'form_processing'})
    
    async def _build_structure(self, agent, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build document structure with enhanced content integration."""
        try:
            # Prepare content data from all processing results
            content_data = {}
            
            for agent_name, result in processed_results.items():
                if result.get('success'):
                    content_data[agent_name] = result.get('result', {})
            
            # Process with structure agent
            structure_result = await agent.process(content_data)
            
            # Validate final markdown
            if structure_result.get('markdown'):
                validation = self.content_validator.validate_markdown(structure_result['markdown'])
                structure_result['markdown_validation'] = validation.to_dict()
                
                # Apply final cleaning
                cleaned_markdown = self.content_cleaner.fix_common_formatting_issues(
                    structure_result['markdown']
                )
                structure_result['cleaned_markdown'] = cleaned_markdown
            
            return structure_result
            
        except Exception as e:
            return agent.error_handler.handle_error(e, context={'operation': 'structure_building'})
    
    async def _validate_quality(self, agent, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document quality with comprehensive checks."""
        try:
            # Get structure result for validation
            structure_result = processed_results.get('StructureBuilderAgent', {})
            
            if not structure_result.get('success'):
                return {'success': False, 'error': 'No structure result to validate'}
            
            # Process with quality agent
            quality_result = agent.process(structure_result)
            
            return quality_result
            
        except Exception as e:
            return agent.error_handler.handle_error(e, context={'operation': 'quality_validation'})
    
    async def _generate_final_output(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final processing output."""
        
        # Get structure result (main output)
        structure_result = results.get('StructureBuilderAgent', {})
        
        if not structure_result.get('success'):
            return {
                'success': False,
                'error': 'Document structure building failed',
                'partial_results': results
            }
        
        # Use cleaned markdown if available
        final_markdown = structure_result.get('cleaned_markdown', 
                                            structure_result.get('markdown', ''))
        
        # Compile metadata
        metadata = {
            'agents_used': list(results.keys()),
            'successful_agents': [name for name, result in results.items() 
                                 if result.get('success')],
            'failed_agents': [name for name, result in results.items() 
                             if not result.get('success')],
            'total_processing_time': sum(
                result.get('processing_time', 0) 
                for result in results.values() 
                if isinstance(result.get('processing_time'), (int, float))
            )
        }
        
        return {
            'success': True,
            'markdown': final_markdown,
            'metadata': metadata,
            'agent_results': results
        }
    
    def _get_processing_insights(self) -> Dict[str, Any]:
        """Get insights from agent processing."""
        return self.memory_manager.get_cross_agent_insights()
    
    def get_agent_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report for all agents."""
        report = {}
        
        for agent in self.pipeline:
            status = agent.get_agent_status()
            report[agent.name] = status
        
        return report


# Example usage function
async def main():
    """Example usage of enhanced PPARSER system."""
    
    # Initialize configuration
    config = Config()  # Your existing config class
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Create processor
    processor = EnhancedPPARSERProcessor(config, output_dir)
    
    # Process document
    pdf_path = Path("./test_data/input/example_document.pdf")
    
    if pdf_path.exists():
        result = await processor.process_document(pdf_path)
        
        if result['success']:
            print("✅ Document processed successfully!")
            print(f"Generated {len(result['markdown'])} characters of markdown")
            
            # Save output
            output_file = output_dir / "enhanced_output.md"
            output_file.write_text(result['markdown'])
            print(f"📄 Output saved to {output_file}")
            
            # Print insights
            insights = result.get('processing_insights', {})
            print(f"🔍 Total agents used: {insights.get('total_agents', 0)}")
            print(f"📊 Total memory entries: {insights.get('total_entries', 0)}")
            
        else:
            print("❌ Document processing failed:")
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    else:
        print(f"❌ PDF file not found: {pdf_path}")


if __name__ == "__main__":
    asyncio.run(main())
