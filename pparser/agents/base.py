"""
Base agent class with enhanced functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from ..utils import logger
from .mixins import LLMInteractionMixin, ContentFormattingMixin, ValidationMixin
from .error_handling import ErrorHandler, with_error_handling, ErrorSeverity
from .memory_system import AgentMemory, MemoryType


class BaseAgent(ABC, LLMInteractionMixin, ContentFormattingMixin, ValidationMixin):
    """Enhanced base class for all PPARSER agents with improved functionality."""
    
    def __init__(self, config, name: str = None, role: str = None, 
                 temperature: Optional[float] = None, max_entries: int = 100):
        super().__init__()
        
        self.config = config
        self.name = name or self.__class__.__name__
        self.role = role or "assistant"
        self.logger = logger
        
        # Initialize enhanced components
        self.error_handler = ErrorHandler(self.name)
        self.memory = AgentMemory(self.name, max_entries)
        
        # Initialize OpenAI LLM if config is available
        if self.config and hasattr(self.config, 'openai_api_key'):
            self.llm = ChatOpenAI(
                model=getattr(self.config, 'openai_model', 'gpt-3.5-turbo'),
                temperature=temperature or getattr(self.config, 'openai_temperature', 0.7),
                max_tokens=getattr(self.config, 'openai_max_tokens', 4000),
                openai_api_key=self.config.openai_api_key
            )
        else:
            self.llm = None
        
        # Agent state for backward compatibility
        self.state: Dict[str, Any] = {}
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process input data and return results."""
        pass
    
    def _create_messages(self, system_prompt: str, user_content: str) -> List[BaseMessage]:
        """Create messages using enhanced mixin functionality."""
        return self.create_standard_messages(system_prompt, user_content)
    
    def _invoke_llm(self, messages: List[BaseMessage], 
                   parse_json: bool = False) -> Optional[str]:
        """Invoke LLM with enhanced error handling and memory."""
        try:
            # Record interaction attempt
            prompt_summary = messages[0].content[:100] if messages else "No prompt"
            
            result = self.invoke_llm_with_retry(messages, parse_json=parse_json)
            
            if result:
                # Record successful interaction
                self.memory.add_interaction(
                    prompt=prompt_summary,
                    response=result[:200],
                    success=True,
                    metadata={'parse_json': parse_json}
                )
            else:
                # Record failed interaction
                self.memory.add_interaction(
                    prompt=prompt_summary,
                    response="Failed to get response",
                    success=False,
                    metadata={'parse_json': parse_json}
                )
            
            return result
            
        except Exception as e:
            error_info = self.error_handler.handle_error(
                e, 
                context={'operation': 'llm_invocation'},
                severity=ErrorSeverity.HIGH
            )
            
            # Record error in memory
            self.memory.add_entry(
                MemoryType.ERROR,
                content={'error': str(e), 'operation': 'llm_invocation'},
                importance=6
            )
            
            return None
    
    @with_error_handling(fallback_value={})
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        memory_summary = self.memory.get_memory_summary()
        error_stats = self.error_handler.get_error_stats()
        
        return {
            'agent_name': self.name,
            'role': self.role,
            'memory_summary': memory_summary,
            'error_stats': error_stats,
            'state_keys': list(self.state.keys()),
            'llm_available': self.llm is not None
        }
    
    def add_context(self, context_type: str, data: Dict[str, Any]) -> str:
        """Add context to agent memory."""
        return self.memory.add_context(context_type, data)
    
    def record_result(self, operation: str, result: Any, 
                     processing_time: float = None) -> str:
        """Record operation result in memory."""
        return self.memory.add_result(operation, result, processing_time)
    
    # Legacy methods for backward compatibility
    def add_to_memory(self, message: BaseMessage):
        """Legacy method - now uses enhanced memory system."""
        content = {'message_type': type(message).__name__, 'content': message.content}
        self.memory.add_entry(MemoryType.INTERACTION, content, importance=2)
    
    def update_state(self, key: str, value: Any):
        """Update agent state."""
        self.state[key] = value
        
        # Also record in memory for better tracking
        self.memory.add_entry(
            MemoryType.METADATA,
            content={'state_update': {key: str(value)[:100]}},
            tags=['state'],
            importance=1
        )
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from agent state."""
        return self.state.get(key, default)
