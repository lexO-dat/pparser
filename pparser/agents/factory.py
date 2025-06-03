"""
Agent factory for creating and managing agent instances.
"""

from typing import Dict, Any, Optional, Type, List, List
from pathlib import Path

from .base import BaseAgent
from .text_agent import TextAnalysisAgent, ContentCleaningAgent
from .image_agent import ImageAnalysisAgent
from .table_agent import TableAnalysisAgent, TablePositionAgent
from .formula_agent import FormulaAnalysisAgent, FormulaFormattingAgent
from .form_agent import FormAnalysisAgent
from .structure_agent import StructureBuilderAgent
from .quality_agent import QualityValidatorAgent
from .config_manager import AgentConfigManager
from .memory_system import MemoryManager


class AgentFactory:
    """Factory for creating and managing agent instances."""
    
    # Registry of available agent classes
    AGENT_REGISTRY = {
        'text': TextAnalysisAgent,
        'text_cleaning': ContentCleaningAgent,
        'image': ImageAnalysisAgent,
        'table': TableAnalysisAgent,
        'table_position': TablePositionAgent,
        'formula': FormulaAnalysisAgent,
        'formula_formatting': FormulaFormattingAgent,
        'form': FormAnalysisAgent,
        'structure': StructureBuilderAgent,
        'quality': QualityValidatorAgent
    }
    
    def __init__(self, global_config):
        """Initialize factory with global configuration."""
        self.global_config = global_config
        self.config_manager = AgentConfigManager(global_config)
        self.memory_manager = MemoryManager()
        self._agent_instances: Dict[str, BaseAgent] = {}
    
    def create_agent(self, agent_type: str, 
                    config_overrides: Optional[Dict[str, Any]] = None,
                    output_dir: Optional[Path] = None,
                    instance_id: Optional[str] = None) -> BaseAgent:
        """Create agent instance with proper configuration."""
        
        if agent_type not in self.AGENT_REGISTRY:
            available_types = ', '.join(self.AGENT_REGISTRY.keys())
            raise ValueError(f"Unknown agent type '{agent_type}'. Available types: {available_types}")
        
        agent_class = self.AGENT_REGISTRY[agent_type]
        
        # Get agent configuration
        agent_config = self.config_manager.get_agent_config(
            agent_class.__name__, 
            overrides=config_overrides
        )
        
        # Create instance with appropriate parameters
        init_kwargs = {'config': self.global_config}
        
        # Add output_dir for agents that need it
        if agent_type in ['table', 'image']:
            init_kwargs['output_dir'] = output_dir
        
        # Create agent instance
        agent = agent_class(**init_kwargs)
        
        # Set up enhanced memory if not already done
        if not hasattr(agent, 'memory') or agent.memory is None:
            agent.memory = self.memory_manager.get_memory(agent.name)
        
        # Store instance if ID provided
        if instance_id:
            self._agent_instances[instance_id] = agent
        
        return agent
    
    def get_agent(self, instance_id: str) -> Optional[BaseAgent]:
        """Get existing agent instance by ID."""
        return self._agent_instances.get(instance_id)
    
    def create_agent_pipeline(self, agent_types: List[str], 
                             shared_config: Optional[Dict[str, Any]] = None,
                             output_dir: Optional[Path] = None) -> List[BaseAgent]:
        """Create a pipeline of agents for processing."""
        agents = []
        
        for agent_type in agent_types:
            agent = self.create_agent(
                agent_type, 
                config_overrides=shared_config,
                output_dir=output_dir
            )
            agents.append(agent)
        
        return agents
    
    def get_default_processing_pipeline(self, output_dir: Optional[Path] = None) -> List[BaseAgent]:
        """Get default agent pipeline for PDF processing."""
        pipeline_types = [
            'text',
            'image', 
            'table',
            'formula',
            'form',
            'structure',
            'quality'
        ]
        
        return self.create_agent_pipeline(pipeline_types, output_dir=output_dir)
    
    def create_specialized_pipeline(self, content_focus: str, 
                                  output_dir: Optional[Path] = None) -> List[BaseAgent]:
        """Create specialized pipeline based on content focus."""
        
        pipelines = {
            'academic': ['text', 'formula', 'table', 'image', 'structure', 'quality'],
            'business': ['text', 'table', 'form', 'image', 'structure', 'quality'],
            'technical': ['text', 'formula', 'table', 'image', 'structure', 'quality'],
            'forms': ['text', 'form', 'table', 'structure', 'quality'],
            'minimal': ['text', 'structure']
        }
        
        if content_focus not in pipelines:
            available_focuses = ', '.join(pipelines.keys())
            raise ValueError(f"Unknown content focus '{content_focus}'. Available: {available_focuses}")
        
        return self.create_agent_pipeline(pipelines[content_focus], output_dir=output_dir)
    
    def get_agent_info(self, agent_type: str) -> Dict[str, Any]:
        """Get information about an agent type."""
        if agent_type not in self.AGENT_REGISTRY:
            return {}
        
        agent_class = self.AGENT_REGISTRY[agent_type]
        config = self.config_manager.get_agent_config(agent_class.__name__)
        
        return {
            'type': agent_type,
            'class_name': agent_class.__name__,
            'description': agent_class.__doc__ or "No description available",
            'default_config': {
                'name': config.name,
                'role': config.role,
                'temperature': config.temperature,
                'max_tokens': config.max_tokens
            }
        }
    
    def create_standard_pipeline(self) -> List[BaseAgent]:
        """Create standard processing pipeline."""
        return self.create_agent_pipeline([
            'text', 'image', 'table', 'formula', 'form', 'structure', 'quality'
        ])
    
    def create_academic_pipeline(self) -> List[BaseAgent]:
        """Create academic document processing pipeline."""
        return self.create_agent_pipeline([
            'text', 'formula', 'table', 'image', 'structure', 'quality'
        ])
    
    def create_technical_pipeline(self) -> List[BaseAgent]:
        """Create technical document processing pipeline."""
        return self.create_agent_pipeline([
            'text', 'formula', 'table', 'image', 'form', 'structure', 'quality'
        ])
    
    def create_fast_pipeline(self) -> List[BaseAgent]:
        """Create fast processing pipeline with minimal agents."""
        return self.create_agent_pipeline([
            'text', 'structure'
        ])
    
    def create_enhanced_processor(self, **kwargs):
        """Create enhanced processor with specified capabilities."""
        from .enhanced_processor import EnhancedProcessor
        
        return EnhancedProcessor(
            agent_factory=self,
            **kwargs
        )
    
    def create_enhanced_batch_processor(self, **kwargs):
        """Create enhanced batch processor."""
        from .enhanced_batch_processor import EnhancedBatchProcessor
        
        return EnhancedBatchProcessor(
            agent_factory=self,
            **kwargs
        )
    
    def list_available_agents(self) -> List[str]:
        """List all available agent types."""
        return list(self.AGENT_REGISTRY.keys())
    
    async def test_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Test functionality of all agents."""
        results = {}
        
        for agent_type in self.AGENT_REGISTRY.keys():
            try:
                agent = self.create_agent(agent_type)
                # Simple test - check if agent initializes properly
                test_result = {
                    'success': True,
                    'message': f"Agent {agent_type} initialized successfully",
                    'agent_name': agent.name if hasattr(agent, 'name') else 'Unknown'
                }
            except Exception as e:
                test_result = {
                    'success': False,
                    'message': f"Failed to initialize: {str(e)}"
                }
            
            results[agent_type] = test_result
        
        return results
    
    def cleanup_instances(self):
        """Clean up stored agent instances."""
        self._agent_instances.clear()
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights from all agent memories."""
        return self.memory_manager.get_cross_agent_insights()
    
    def validate_pipeline(self, agent_types: List[str]) -> Dict[str, Any]:
        """Validate that a pipeline configuration is valid."""
        issues = []
        warnings = []
        
        # Check if all agent types exist
        for agent_type in agent_types:
            if agent_type not in self.AGENT_REGISTRY:
                issues.append(f"Unknown agent type: {agent_type}")
        
        # Check for recommended order
        recommended_order = ['text', 'image', 'table', 'formula', 'form', 'structure', 'quality']
        for i, agent_type in enumerate(agent_types):
            if agent_type in recommended_order:
                recommended_index = recommended_order.index(agent_type)
                current_position = i
                
                # Check if any earlier agent in recommended order comes after this one
                for j in range(current_position + 1, len(agent_types)):
                    later_agent = agent_types[j]
                    if later_agent in recommended_order:
                        later_recommended_index = recommended_order.index(later_agent)
                        if later_recommended_index < recommended_index:
                            warnings.append(f"'{later_agent}' typically runs before '{agent_type}'")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'agent_count': len(agent_types)
        }
