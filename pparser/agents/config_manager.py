"""
Agent configuration management for consistent setup across all agents.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI


@dataclass
class AgentConfig:
    """Configuration container for agent settings."""
    name: str
    role: str
    temperature: float = 0.7
    max_tokens: int = 4000
    model: str = "gpt-3.5-turbo"
    timeout: int = 30
    max_retries: int = 3
    
    # Specialized settings
    chunk_size: Optional[int] = None
    enable_memory: bool = True
    enable_streaming: bool = False


class AgentConfigManager:
    """Centralized configuration management for all agents."""
    
    # Default configurations for each agent type
    DEFAULT_CONFIGS = {
        'TextAnalysisAgent': AgentConfig(
            name="TextAnalysisAgent",
            role="Analyze and structure text content from PDF pages",
            temperature=0.1,
            chunk_size=2000
        ),
        'ImageAnalysisAgent': AgentConfig(
            name="ImageAnalysisAgent", 
            role="Analyze images and generate descriptions",
            temperature=0.3
        ),
        'TableAnalysisAgent': AgentConfig(
            name="TableAnalysisAgent",
            role="Analyze and format tables for Markdown conversion",
            temperature=0.7
        ),
        'FormulaAnalysisAgent': AgentConfig(
            name="FormulaAnalysisAgent",
            role="Analyze and convert mathematical formulas to Markdown",
            temperature=0.0
        ),
        'FormAnalysisAgent': AgentConfig(
            name="FormAnalysisAgent",
            role="Analyze forms and convert to interactive Markdown",
            temperature=0.7
        ),
        'StructureBuilderAgent': AgentConfig(
            name="StructureBuilderAgent",
            role="Build document structure and generate Markdown",
            temperature=0.1,
            max_tokens=6000
        ),
        'QualityValidatorAgent': AgentConfig(
            name="QualityValidatorAgent",
            role="Validate and improve document quality",
            temperature=0.2
        )
    }
    
    def __init__(self, global_config):
        """Initialize with global configuration object."""
        self.global_config = global_config
    
    def get_agent_config(self, agent_type: str, 
                        overrides: Optional[Dict[str, Any]] = None) -> AgentConfig:
        """Get configuration for specific agent type with optional overrides."""
        if agent_type not in self.DEFAULT_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        config = self.DEFAULT_CONFIGS[agent_type]
        
        # Apply global overrides
        if hasattr(self.global_config, 'openai_model'):
            config.model = self.global_config.openai_model
        if hasattr(self.global_config, 'openai_max_tokens'):
            config.max_tokens = self.global_config.openai_max_tokens
        if hasattr(self.global_config, 'openai_temperature'):
            config.temperature = self.global_config.openai_temperature
        
        # Apply local overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def create_llm_instance(self, agent_config: AgentConfig) -> ChatOpenAI:
        """Create LLM instance with agent-specific configuration."""
        return ChatOpenAI(
            model=agent_config.model,
            temperature=agent_config.temperature,
            max_tokens=agent_config.max_tokens,
            openai_api_key=self.global_config.openai_api_key,
            request_timeout=agent_config.timeout
        )
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent types."""
        return list(self.DEFAULT_CONFIGS.keys())
    
    def validate_agent_config(self, config: AgentConfig) -> bool:
        """Validate agent configuration."""
        if not config.name or not config.role:
            return False
        if config.temperature < 0 or config.temperature > 2:
            return False
        if config.max_tokens < 100 or config.max_tokens > 8000:
            return False
        return True
