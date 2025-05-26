"""
Unit tests for configuration management.
"""
import pytest
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from pparser.config import Config


class TestConfig:
    """Test configuration management."""
    
    def test_config_default_values(self):
        """Test that config uses default values when not specified."""
        config = Config(openai_api_key="test-key")
        
        assert config.openai_api_key == "test-key"
        assert config.openai_model == "gpt-4o-mini"
        assert config.temperature == 0.1
        assert config.max_tokens == 4096
        assert config.max_concurrent_pages == 5
        assert config.chunk_size == 2048
        assert config.output_format == "markdown"
        assert config.log_level == "INFO"
    
    def test_config_custom_values(self):
        """Test that config accepts custom values."""
        config = Config(
            openai_api_key="custom-key",
            openai_model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
            max_concurrent_pages=10,
            chunk_size=1024,
            output_format="html",
            log_level="DEBUG"
        )
        
        assert config.openai_api_key == "custom-key"
        assert config.openai_model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.max_concurrent_pages == 10
        assert config.chunk_size == 1024
        assert config.output_format == "html"
        assert config.log_level == "DEBUG"
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'env-key',
        'OPENAI_MODEL': 'gpt-3.5-turbo',
        'OPENAI_TEMPERATURE': '0.7',
        'OPENAI_MAX_TOKENS': '1024',
        'MAX_CONCURRENT_PAGES': '3',
        'CHUNK_SIZE': '512',
        'OUTPUT_FORMAT': 'json',
        'LOG_LEVEL': 'WARNING'
    })
    def test_config_from_environment(self):
        """Test that config loads from environment variables."""
        config = Config.from_env()
        
        assert config.openai_api_key == "env-key"
        assert config.openai_model == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.max_concurrent_pages == 3
        assert config.chunk_size == 512
        assert config.output_format == "json"
        assert config.log_level == "WARNING"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_config_missing_api_key(self):
        """Test that config raises error when API key is missing."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            Config.from_env()
    
    def test_config_output_dir_property(self, temp_dir):
        """Test output_dir property."""
        config = Config(openai_api_key="test-key", output_dir=temp_dir)
        assert config.output_dir == temp_dir
    
    def test_config_assets_dir_property(self, temp_dir):
        """Test assets_dir property."""
        config = Config(openai_api_key="test-key", output_dir=temp_dir)
        assets_dir = config.assets_dir
        assert assets_dir == temp_dir / "assets"
    
    def test_config_dict_conversion(self):
        """Test converting config to dictionary."""
        config = Config(
            openai_api_key="test-key",
            openai_model="gpt-4",
            temperature=0.3
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["openai_model"] == "gpt-4"
        assert config_dict["temperature"] == 0.3
        # API key should not be included for security
        assert "openai_api_key" not in config_dict
    
    def test_config_validation_temperature(self):
        """Test temperature validation."""
        # Valid temperatures
        Config(openai_api_key="test-key", temperature=0.0)
        Config(openai_api_key="test-key", temperature=1.0)
        Config(openai_api_key="test-key", temperature=0.5)
        
        # Invalid temperatures should use default
        config = Config(openai_api_key="test-key", temperature=-0.1)
        assert config.temperature == 0.1  # Default value
        
        config = Config(openai_api_key="test-key", temperature=1.1)
        assert config.temperature == 0.1  # Default value
    
    def test_config_validation_max_tokens(self):
        """Test max_tokens validation."""
        # Valid values
        Config(openai_api_key="test-key", max_tokens=1)
        Config(openai_api_key="test-key", max_tokens=4096)
        
        # Invalid values should use default
        config = Config(openai_api_key="test-key", max_tokens=0)
        assert config.max_tokens == 4096  # Default value
        
        config = Config(openai_api_key="test-key", max_tokens=-100)
        assert config.max_tokens == 4096  # Default value
    
    def test_config_validation_concurrent_pages(self):
        """Test max_concurrent_pages validation."""
        # Valid values
        Config(openai_api_key="test-key", max_concurrent_pages=1)
        Config(openai_api_key="test-key", max_concurrent_pages=10)
        
        # Invalid values should use default
        config = Config(openai_api_key="test-key", max_concurrent_pages=0)
        assert config.max_concurrent_pages == 5  # Default value
    
    def test_config_validation_chunk_size(self):
        """Test chunk_size validation."""
        # Valid values
        Config(openai_api_key="test-key", chunk_size=100)
        Config(openai_api_key="test-key", chunk_size=4096)
        
        # Invalid values should use default
        config = Config(openai_api_key="test-key", chunk_size=0)
        assert config.chunk_size == 2048  # Default value
    
    def test_config_str_representation(self):
        """Test string representation of config."""
        config = Config(openai_api_key="test-key", openai_model="gpt-4")
        str_repr = str(config)
        
        assert "Config" in str_repr
        assert "gpt-4" in str_repr
        # API key should be masked
        assert "test-key" not in str_repr or "***" in str_repr
