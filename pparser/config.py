"""
Configuration management for PPARSER system
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration class for PPARSER system"""
    
    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 4096
    
    # Processing settings
    max_concurrent_pages: int = 5
    chunk_size: int = 2048
    output_format: str = "markdown"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "pparser.log"
    
    # Directories
    temp_dir: Path = Path("temp")
    output_dir: Path = Path("output")
    
    def __init__(self, 
                 openai_api_key: str = None,
                 openai_model: str = None,
                 temperature: float = None,  # Support both names
                 openai_temperature: float = None,
                 max_tokens: int = None,  # Support both names
                 openai_max_tokens: int = None,
                 max_concurrent_pages: int = None,
                 chunk_size: int = None,
                 output_format: str = None,
                 log_level: str = None,
                 log_file: str = None,
                 temp_dir: str = None,
                 output_dir: str = None):
        """Initialize config with optional parameters, falling back to environment variables"""
        
        # OpenAI settings with fallbacks
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Handle temperature with dual naming support
        temp_value = temperature or openai_temperature or float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.openai_temperature = temp_value
        self.temperature = temp_value  # Alias for tests
        
        # Handle max_tokens with dual naming support
        tokens_value = max_tokens or openai_max_tokens or int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
        self.openai_max_tokens = tokens_value
        self.max_tokens = tokens_value  # Alias for tests
        
        # Processing settings
        self.max_concurrent_pages = max_concurrent_pages if max_concurrent_pages is not None else int(os.getenv("MAX_CONCURRENT_PAGES", "5"))
        self.chunk_size = chunk_size if chunk_size is not None else int(os.getenv("CHUNK_SIZE", "2048"))
        self.output_format = output_format or os.getenv("OUTPUT_FORMAT", "markdown")
        
        # Logging
        self.log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
        self.log_file = log_file or os.getenv("LOG_FILE", "pparser.log")
        
        # Directories
        self.temp_dir = Path(temp_dir) if temp_dir else Path(os.getenv("TEMP_DIR", "temp"))
        self.output_dir = Path(output_dir) if output_dir else Path(os.getenv("OUTPUT_DIR", "output"))
        
        # Create directories if they don't exist
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate configuration
        self._validate()
    
    @property
    def assets_dir(self) -> Path:
        """Get assets directory path"""
        assets_path = self.output_dir / "assets"
        assets_path.mkdir(exist_ok=True)
        return assets_path
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables only"""
        return cls()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "openai_api_key": "***" if self.openai_api_key else "",
            "openai_model": self.openai_model,
            "openai_temperature": self.openai_temperature,
            "openai_max_tokens": self.openai_max_tokens,
            "max_concurrent_pages": self.max_concurrent_pages,
            "chunk_size": self.chunk_size,
            "output_format": self.output_format,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "temp_dir": str(self.temp_dir),
            "output_dir": str(self.output_dir)
        }
    
    def _validate(self):
        """Validate configuration values"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        if not 0.0 <= self.openai_temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        if self.openai_max_tokens < 1:
            raise ValueError("Max tokens must be positive")
        
        if self.max_concurrent_pages < 1:
            raise ValueError("Max concurrent pages must be positive")
        
        if self.chunk_size < 1:
            raise ValueError("Chunk size must be positive")
    
    def __str__(self) -> str:
        """String representation with masked API key"""
        return (f"Config(openai_api_key='***', openai_model='{self.openai_model}', "
                f"openai_temperature={self.openai_temperature}, openai_max_tokens={self.openai_max_tokens}, "
                f"max_concurrent_pages={self.max_concurrent_pages}, chunk_size={self.chunk_size}, "
                f"output_format='{self.output_format}', log_level='{self.log_level}', "
                f"log_file='{self.log_file}', temp_dir={self.temp_dir}, output_dir={self.output_dir})")


# Global configuration instance
config = Config()
