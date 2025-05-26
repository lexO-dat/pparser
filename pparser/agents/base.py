"""
Base agent class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from ..utils import logger


class BaseAgent(ABC):
    """Base class for all PPARSER agents"""
    
    def __init__(self, config, name: str = None, role: str = None, temperature: Optional[float] = None):
        self.config = config
        self.name = name or self.__class__.__name__
        self.role = role or "assistant"
        self.logger = logger
        
        # Initialize OpenAI LLM if config is available
        if self.config and hasattr(self.config, 'openai_api_key'):
            self.llm = ChatOpenAI(
                model=self.config.openai_model,
                temperature=temperature or self.config.openai_temperature,
                max_tokens=self.config.openai_max_tokens,
                openai_api_key=self.config.openai_api_key
            )
        else:
            self.llm = None
        
        # Agent memory and state
        # TODO: Implement a better memory management system (like using postgres or buffer memory)
        self.memory: List[BaseMessage] = []
        self.state: Dict[str, Any] = {}
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    def _create_messages(self, system_prompt: str, user_content: str) -> List[BaseMessage]:
        """LLM interaction Messages"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        return messages
    
    def _invoke_llm(self, messages: List[BaseMessage]) -> str:
        """Invoke LLM"""
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"LLM invocation failed for {self.name}: {e}")
            return ""
    
    # TODO: as is written above, i have to implement a better memory system
    def add_to_memory(self, message: BaseMessage):
        """Add message to agent memory"""
        self.memory.append(message)
        
        # Keep memory within reasonable limits
        if len(self.memory) > 20:
            self.memory = self.memory[-15:]  # Keep last 15 messages
    
    def update_state(self, key: str, value: Any):
        """Update agent state"""
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from agent state"""
        return self.state.get(key, default)
