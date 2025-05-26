"""
Unit tests for base extractor and agent classes.
"""
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from pparser.extractors.base import BaseExtractor
from pparser.agents.base import BaseAgent
from pparser.config import Config


class TestBaseExtractor:
    """Test the BaseExtractor abstract class."""
    
    def test_base_extractor_cannot_be_instantiated(self):
        """Test that BaseExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseExtractor(config=Mock())
    
    def test_base_extractor_subclass_implementation(self, test_config):
        """Test that BaseExtractor subclass works correctly."""
        
        class TestExtractor(BaseExtractor):
            def extract(self, pdf_path, output_dir):
                return {"test": "data"}
        
        extractor = TestExtractor(test_config)
        assert extractor.config == test_config
        
        result = extractor.extract("test.pdf", "output")
        assert result == {"test": "data"}
    
    def test_base_extractor_requires_extract_method(self, test_config):
        """Test that BaseExtractor subclass must implement extract method."""
        
        class IncompleteExtractor(BaseExtractor):
            pass
        
        with pytest.raises(TypeError):
            IncompleteExtractor(test_config)


class TestBaseAgent:
    """Test the BaseAgent abstract class."""
    
    def test_base_agent_cannot_be_instantiated(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent(config=Mock())
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_base_agent_subclass_implementation(self, mock_chat_openai, test_config):
        """Test that BaseAgent subclass works correctly."""
        
        # Mock the ChatOpenAI instance
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        class TestAgent(BaseAgent):
            def analyze(self, data):
                return {"analyzed": True, "data": data}
        
        agent = TestAgent(test_config)
        assert agent.config == test_config
        assert agent.llm == mock_llm
        
        # Verify ChatOpenAI was initialized with correct parameters
        mock_chat_openai.assert_called_once_with(
            api_key=test_config.openai_api_key,
            model=test_config.openai_model,
            temperature=test_config.temperature,
            max_tokens=test_config.max_tokens
        )
        
        result = agent.analyze({"test": "input"})
        assert result == {"analyzed": True, "data": {"test": "input"}}
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_base_agent_requires_analyze_method(self, mock_chat_openai, test_config):
        """Test that BaseAgent subclass must implement analyze method."""
        
        class IncompleteAgent(BaseAgent):
            pass
        
        with pytest.raises(TypeError):
            IncompleteAgent(test_config)
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_base_agent_invoke_llm(self, mock_chat_openai, test_config):
        """Test the _invoke_llm method."""
        
        # Mock the ChatOpenAI instance and its response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response.additional_kwargs = {}
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        class TestAgent(BaseAgent):
            def analyze(self, data):
                return self._invoke_llm("Test prompt", {"input": data})
        
        agent = TestAgent(test_config)
        result = agent.analyze("test data")
        
        # Verify the LLM was invoked correctly
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Test prompt" in str(call_args)
        
        assert result == "Test response"
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_base_agent_invoke_llm_with_error_handling(self, mock_chat_openai, test_config):
        """Test _invoke_llm with error handling."""
        
        # Mock the ChatOpenAI instance to raise an exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API Error")
        mock_chat_openai.return_value = mock_llm
        
        class TestAgent(BaseAgent):
            def analyze(self, data):
                return self._invoke_llm("Test prompt", {"input": data})
        
        agent = TestAgent(test_config)
        
        # Should handle the exception gracefully
        result = agent.analyze("test data")
        assert result is None or isinstance(result, str)
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_base_agent_invoke_llm_with_json_response(self, mock_chat_openai, test_config):
        """Test _invoke_llm with JSON response parsing."""
        
        # Mock the ChatOpenAI instance and its response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"result": "success", "data": ["item1", "item2"]}'
        mock_response.additional_kwargs = {}
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        class TestAgent(BaseAgent):
            def analyze(self, data):
                return self._invoke_llm("Test prompt", {"input": data}, parse_json=True)
        
        agent = TestAgent(test_config)
        result = agent.analyze("test data")
        
        assert isinstance(result, dict)
        assert result["result"] == "success"
        assert result["data"] == ["item1", "item2"]
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_base_agent_invoke_llm_invalid_json(self, mock_chat_openai, test_config):
        """Test _invoke_llm with invalid JSON response."""
        
        # Mock the ChatOpenAI instance with invalid JSON response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Invalid JSON response"
        mock_response.additional_kwargs = {}
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        class TestAgent(BaseAgent):
            def analyze(self, data):
                return self._invoke_llm("Test prompt", {"input": data}, parse_json=True)
        
        agent = TestAgent(test_config)
        result = agent.analyze("test data")
        
        # Should return None or handle gracefully when JSON parsing fails
        assert result is None or isinstance(result, str)
