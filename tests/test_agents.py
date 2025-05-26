"""
Unit tests for agent functionality.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from pparser.agents.text_agent import TextAnalysisAgent
from pparser.agents.image_agent import ImageAnalysisAgent
from pparser.agents.table_agent import TableAnalysisAgent
from pparser.agents.structure_agent import StructureBuilderAgent
from pparser.agents.quality_agent import QualityValidatorAgent


class TestTextAnalysisAgent:
    """Test the TextAnalysisAgent class."""
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_text_agent_initialization(self, mock_chat_openai, test_config):
        """Test TextAnalysisAgent initialization."""
        agent = TextAnalysisAgent(test_config)
        assert agent.config == test_config
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_analyze_text_content(self, mock_chat_openai, test_config, sample_text_content):
        """Test text content analysis."""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "structure": {
                "title": "Enhanced Document Title",
                "sections": ["Introduction", "Main Content", "Conclusion"]
            },
            "improvements": {
                "cleaned_text": "This is enhanced text content.",
                "headings": [{"text": "Introduction", "level": 1}]
            }
        })
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        agent = TextAnalysisAgent(test_config)
        result = agent.analyze(sample_text_content)
        
        assert isinstance(result, dict)
        assert 'structure' in result
        assert 'improvements' in result
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_analyze_with_error_handling(self, mock_chat_openai, test_config):
        """Test error handling in text analysis."""
        # Mock LLM to raise an exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API Error")
        mock_chat_openai.return_value = mock_llm
        
        agent = TextAnalysisAgent(test_config)
        result = agent.analyze({"pages": []})
        
        # Should handle error gracefully
        assert result is not None


class TestImageAnalysisAgent:
    """Test the ImageAnalysisAgent class."""
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_image_agent_initialization(self, mock_chat_openai, test_config):
        """Test ImageAnalysisAgent initialization."""
        agent = ImageAnalysisAgent(test_config)
        assert agent.config == test_config
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_analyze_image_content(self, mock_chat_openai, test_config, sample_image_data):
        """Test image content analysis."""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "enhanced_descriptions": [
                {
                    "image_path": "test_image.png",
                    "description": "Enhanced description of the image",
                    "alt_text": "Alternative text for accessibility",
                    "category": "diagram"
                }
            ],
            "positioning": [
                {
                    "image_path": "test_image.png",
                    "placement": "center",
                    "caption": "Figure 1: Test Image"
                }
            ]
        })
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        agent = ImageAnalysisAgent(test_config)
        result = agent.analyze(sample_image_data)
        
        assert isinstance(result, dict)
        assert 'enhanced_descriptions' in result
        assert 'positioning' in result


class TestTableAnalysisAgent:
    """Test the TableAnalysisAgent class."""
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_table_agent_initialization(self, mock_chat_openai, test_config):
        """Test TableAnalysisAgent initialization."""
        agent = TableAnalysisAgent(test_config)
        assert agent.config == test_config
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_analyze_table_content(self, mock_chat_openai, test_config, sample_table_data):
        """Test table content analysis."""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "enhanced_tables": [
                {
                    "caption": "Enhanced Table Caption",
                    "headers": ["Column 1", "Column 2"],
                    "formatting": {
                        "alignment": ["left", "right"],
                        "style": "bordered"
                    }
                }
            ],
            "improvements": {
                "data_types": ["text", "number"],
                "summary": "Table contains statistical data"
            }
        })
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        agent = TableAnalysisAgent(test_config)
        result = agent.analyze(sample_table_data)
        
        assert isinstance(result, dict)
        assert 'enhanced_tables' in result
        assert 'improvements' in result


class TestStructureBuilderAgent:
    """Test the StructureBuilderAgent class."""
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_structure_agent_initialization(self, mock_chat_openai, test_config):
        """Test StructureBuilderAgent initialization."""
        agent = StructureBuilderAgent(test_config)
        assert agent.config == test_config
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_build_document_structure(self, mock_chat_openai, test_config):
        """Test document structure building."""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "document_structure": {
                "title": "Complete Document Title",
                "sections": [
                    {
                        "title": "Introduction",
                        "content": "Introduction content with images and tables",
                        "subsections": []
                    }
                ]
            },
            "markdown_structure": {
                "toc": "# Table of Contents\n1. Introduction",
                "content_order": ["text", "images", "tables", "formulas"]
            }
        })
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Sample consolidated data
        consolidated_data = {
            "text": {"structure": {"title": "Test Doc"}},
            "images": {"images": []},
            "tables": {"tables": []},
            "formulas": {"formulas": []},
            "forms": {"forms": []}
        }
        
        agent = StructureBuilderAgent(test_config)
        result = agent.analyze(consolidated_data)
        
        assert isinstance(result, dict)
        assert 'document_structure' in result
        assert 'markdown_structure' in result


class TestQualityValidatorAgent:
    """Test the QualityValidatorAgent class."""
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_quality_agent_initialization(self, mock_chat_openai, test_config):
        """Test QualityValidatorAgent initialization."""
        agent = QualityValidatorAgent(test_config)
        assert agent.config == test_config
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_validate_document_quality(self, mock_chat_openai, test_config):
        """Test document quality validation."""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "quality_score": 85,
            "dimensions": {
                "structure": 90,
                "completeness": 80,
                "formatting": 85,
                "asset_integrity": 90,
                "readability": 80,
                "accuracy": 85
            },
            "issues": [
                {
                    "type": "formatting",
                    "severity": "minor",
                    "description": "Some tables could have better formatting"
                }
            ],
            "recommendations": [
                "Add more descriptive captions to images",
                "Improve table formatting consistency"
            ]
        })
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        # Sample document data
        document_data = {
            "markdown_content": "# Test Document\n\nThis is test content.",
            "assets": {"images": [], "tables": []},
            "metadata": {"title": "Test Document"}
        }
        
        agent = QualityValidatorAgent(test_config)
        result = agent.analyze(document_data)
        
        assert isinstance(result, dict)
        assert 'quality_score' in result
        assert 'dimensions' in result
        assert 'issues' in result
        assert 'recommendations' in result
        
        # Verify quality score is within valid range
        assert 0 <= result['quality_score'] <= 100
    
    @patch('pparser.agents.base.ChatOpenAI')
    def test_quality_validation_with_issues(self, mock_chat_openai, test_config):
        """Test quality validation with multiple issues."""
        # Mock LLM response with issues
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "quality_score": 65,
            "dimensions": {
                "structure": 70,
                "completeness": 60,
                "formatting": 65,
                "asset_integrity": 70,
                "readability": 60,
                "accuracy": 65
            },
            "issues": [
                {
                    "type": "structure",
                    "severity": "major",
                    "description": "Missing document title"
                },
                {
                    "type": "completeness",
                    "severity": "minor",
                    "description": "Some images lack descriptions"
                }
            ],
            "recommendations": [
                "Add a clear document title",
                "Provide descriptions for all images",
                "Improve overall document structure"
            ]
        })
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        document_data = {
            "markdown_content": "Content without proper structure",
            "assets": {"images": [{"path": "img1.png"}], "tables": []},
            "metadata": {}
        }
        
        agent = QualityValidatorAgent(test_config)
        result = agent.analyze(document_data)
        
        assert result['quality_score'] < 80  # Lower score due to issues
        assert len(result['issues']) >= 2
        assert len(result['recommendations']) >= 3
