"""
Integration tests for the complete PDF parser system.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import shutil
import json

from pparser.processors.pdf_processor import PDFProcessor
from pparser.processors.batch_processor import BatchProcessor
from pparser.config import Config


class TestPDFProcessorIntegration:
    """Integration tests for PDFProcessor."""
    
    @patch('pparser.processors.pdf_processor.PDFWorkflow')
    def test_pdf_processor_initialization(self, mock_workflow, test_config):
        """Test PDFProcessor initialization."""
        processor = PDFProcessor(test_config)
        assert processor.config == test_config
        mock_workflow.assert_called_once_with(test_config)
    
    @patch('pparser.processors.pdf_processor.PDFWorkflow')
    def test_process_single_pdf(self, mock_workflow, test_config, temp_dir, sample_pdf_path):
        """Test processing a single PDF file."""
        # Mock workflow result
        mock_workflow_instance = Mock()
        mock_workflow_instance.process.return_value = {
            "status": "completed",
            "quality_score": 85,
            "final_output": {
                "markdown_file": str(temp_dir / "output.md"),
                "assets_dir": str(temp_dir / "assets")
            },
            "metadata": {
                "pages": 5,
                "processing_time": 30.5
            }
        }
        mock_workflow.return_value = mock_workflow_instance
        
        processor = PDFProcessor(test_config)
        result = processor.process(sample_pdf_path, temp_dir)
        
        assert isinstance(result, dict)
        assert result["status"] == "completed"
        assert result["quality_score"] == 85
        assert "final_output" in result
        assert "metadata" in result
    
    @patch('pparser.processors.pdf_processor.PDFWorkflow')
    def test_process_with_custom_config(self, mock_workflow, temp_dir, sample_pdf_path):
        """Test processing with custom configuration."""
        # Custom config
        custom_config = Config(
            openai_api_key="test-key",
            openai_model="gpt-4",
            output_dir=temp_dir,
            max_concurrent_pages=10
        )
        
        mock_workflow_instance = Mock()
        mock_workflow_instance.process.return_value = {"status": "completed"}
        mock_workflow.return_value = mock_workflow_instance
        
        processor = PDFProcessor(custom_config)
        result = processor.process(sample_pdf_path, temp_dir)
        
        # Verify workflow was called with custom config
        mock_workflow.assert_called_once_with(custom_config)
        assert result["status"] == "completed"
    
    @patch('pparser.processors.pdf_processor.PDFWorkflow')
    def test_process_with_error_handling(self, mock_workflow, test_config, temp_dir):
        """Test error handling during processing."""
        # Mock workflow to raise an exception
        mock_workflow_instance = Mock()
        mock_workflow_instance.process.side_effect = Exception("Processing failed")
        mock_workflow.return_value = mock_workflow_instance
        
        processor = PDFProcessor(test_config)
        result = processor.process("nonexistent.pdf", temp_dir)
        
        assert isinstance(result, dict)
        assert result["status"] == "failed"
        assert "error" in result
    
    @patch('pparser.processors.pdf_processor.PDFWorkflow')
    def test_process_with_quality_threshold(self, mock_workflow, test_config, temp_dir, sample_pdf_path):
        """Test processing with quality threshold checking."""
        # Mock workflow with low quality score
        mock_workflow_instance = Mock()
        mock_workflow_instance.process.return_value = {
            "status": "completed",
            "quality_score": 65,  # Below typical threshold
            "quality_report": {
                "issues": ["Missing document structure"],
                "recommendations": ["Add proper headings"]
            }
        }
        mock_workflow.return_value = mock_workflow_instance
        
        processor = PDFProcessor(test_config)
        result = processor.process(sample_pdf_path, temp_dir, min_quality_score=70)
        
        assert result["status"] == "completed"
        assert result["quality_score"] == 65
        # Should include quality warnings
        assert "quality_report" in result


class TestBatchProcessorIntegration:
    """Integration tests for BatchProcessor."""
    
    def test_batch_processor_initialization(self, test_config):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(test_config)
        assert processor.config == test_config
    
    @pytest.mark.asyncio
    @patch('pparser.processors.batch_processor.BatchWorkflow')
    async def test_process_multiple_files(self, mock_batch_workflow, test_config, temp_dir):
        """Test processing multiple PDF files."""
        # Create sample PDF files
        pdf_files = []
        for i in range(3):
            pdf_path = temp_dir / f"test{i}.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nSample content")
            pdf_files.append(str(pdf_path))
        
        # Mock batch workflow
        mock_workflow_instance = Mock()
        mock_workflow_instance.process_batch = AsyncMock(return_value={
            "summary": {
                "total_files": 3,
                "completed": 3,
                "failed": 0,
                "average_quality": 82.5,
                "total_time": 90.0
            },
            "results": [
                {"file": str(pdf_files[0]), "status": "completed", "quality_score": 85},
                {"file": str(pdf_files[1]), "status": "completed", "quality_score": 80},
                {"file": str(pdf_files[2]), "status": "completed", "quality_score": 82}
            ]
        })
        mock_batch_workflow.return_value = mock_workflow_instance
        
        processor = BatchProcessor(test_config)
        result = await processor.process_batch(pdf_files, temp_dir)
        
        assert isinstance(result, dict)
        assert "summary" in result
        assert "results" in result
        assert result["summary"]["total_files"] == 3
        assert result["summary"]["completed"] == 3
        assert len(result["results"]) == 3
    
    @pytest.mark.asyncio
    @patch('pparser.processors.batch_processor.BatchWorkflow')
    async def test_process_from_directory(self, mock_batch_workflow, test_config, temp_dir):
        """Test processing all PDFs from a directory."""
        # Create a directory with PDF files
        pdf_dir = temp_dir / "pdfs"
        pdf_dir.mkdir()
        
        pdf_files = []
        for i in range(2):
            pdf_path = pdf_dir / f"doc{i}.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nSample content")
            pdf_files.append(pdf_path)
        
        # Add a non-PDF file (should be ignored)
        (pdf_dir / "readme.txt").write_text("Not a PDF")
        
        # Mock batch workflow
        mock_workflow_instance = Mock()
        mock_workflow_instance.process_batch = AsyncMock(return_value={
            "summary": {"total_files": 2, "completed": 2, "failed": 0},
            "results": [
                {"file": str(pdf_files[0]), "status": "completed"},
                {"file": str(pdf_files[1]), "status": "completed"}
            ]
        })
        mock_batch_workflow.return_value = mock_workflow_instance
        
        processor = BatchProcessor(test_config)
        result = await processor.process_directory(pdf_dir, temp_dir)
        
        assert result["summary"]["total_files"] == 2  # Only PDF files
    
    @pytest.mark.asyncio
    @patch('pparser.processors.batch_processor.BatchWorkflow')
    async def test_process_from_file_list(self, mock_batch_workflow, test_config, temp_dir):
        """Test processing from a file list."""
        # Create file list
        pdf_files = []
        for i in range(2):
            pdf_path = temp_dir / f"list_doc{i}.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nSample content")
            pdf_files.append(str(pdf_path))
        
        file_list_path = temp_dir / "files.txt"
        file_list_path.write_text("\n".join(pdf_files))
        
        # Mock batch workflow
        mock_workflow_instance = Mock()
        mock_workflow_instance.process_batch = AsyncMock(return_value={
            "summary": {"total_files": 2, "completed": 2, "failed": 0},
            "results": [
                {"file": pdf_files[0], "status": "completed"},
                {"file": pdf_files[1], "status": "completed"}
            ]
        })
        mock_batch_workflow.return_value = mock_workflow_instance
        
        processor = BatchProcessor(test_config)
        result = await processor.process_file_list(file_list_path, temp_dir)
        
        assert result["summary"]["total_files"] == 2
    
    @pytest.mark.asyncio
    @patch('pparser.processors.batch_processor.BatchWorkflow')
    async def test_batch_processing_with_failures(self, mock_batch_workflow, test_config, temp_dir):
        """Test batch processing with some failures."""
        # Create sample files
        pdf_files = [str(temp_dir / f"test{i}.pdf") for i in range(3)]
        for pdf_file in pdf_files:
            Path(pdf_file).write_bytes(b"%PDF-1.4\nSample content")
        
        # Mock batch workflow with mixed results
        mock_workflow_instance = Mock()
        mock_workflow_instance.process_batch = AsyncMock(return_value={
            "summary": {
                "total_files": 3,
                "completed": 2,
                "failed": 1,
                "average_quality": 75.0,
                "total_time": 120.0
            },
            "results": [
                {"file": pdf_files[0], "status": "completed", "quality_score": 80},
                {"file": pdf_files[1], "status": "failed", "error": "Processing error"},
                {"file": pdf_files[2], "status": "completed", "quality_score": 70}
            ]
        })
        mock_batch_workflow.return_value = mock_workflow_instance
        
        processor = BatchProcessor(test_config)
        result = await processor.process_batch(pdf_files, temp_dir)
        
        assert result["summary"]["completed"] == 2
        assert result["summary"]["failed"] == 1
        
        # Check individual results
        failed_result = next(r for r in result["results"] if r["status"] == "failed")
        assert "error" in failed_result
    
    def test_get_progress_tracking(self, test_config):
        """Test progress tracking functionality."""
        processor = BatchProcessor(test_config)
        
        # Initially no progress
        progress = processor.get_progress()
        assert progress["total"] == 0
        assert progress["completed"] == 0
        
        # Mock workflow with progress
        with patch('pparser.processors.batch_processor.BatchWorkflow') as mock_workflow:
            mock_workflow_instance = Mock()
            mock_workflow_instance.get_progress.return_value = {
                "total": 5,
                "completed": 3,
                "failed": 1,
                "percentage": 80.0
            }
            
            processor.workflow = mock_workflow_instance
            progress = processor.get_progress()
            
            assert progress["total"] == 5
            assert progress["completed"] == 3
            assert progress["failed"] == 1
            assert progress["percentage"] == 80.0


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @patch('pparser.extractors.text.fitz.open')
    @patch('pparser.agents.base.ChatOpenAI')
    def test_complete_pdf_processing_pipeline(self, mock_chat_openai, mock_fitz_open, 
                                            test_config, temp_dir, sample_pdf_path):
        """Test the complete processing pipeline from PDF to Markdown."""
        # Mock PDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.number = 0
        mock_page.get_text.return_value = "Sample document content"
        mock_page.rect = Mock(width=612, height=792)
        mock_page.get_text_dict.return_value = {'blocks': []}
        mock_page.get_images.return_value = []
        
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.metadata = {'title': 'Test Document'}
        mock_fitz_open.return_value = mock_doc
        
        # Mock LLM responses
        mock_llm = Mock()
        mock_responses = [
            # Text analysis response
            Mock(content=json.dumps({
                "structure": {"title": "Test Document", "sections": ["Introduction"]},
                "improvements": {"cleaned_text": "Enhanced content"}
            })),
            # Structure building response
            Mock(content=json.dumps({
                "document_structure": {"title": "Test Document"},
                "markdown_structure": {"content": "# Test Document\nContent"}
            })),
            # Quality validation response
            Mock(content=json.dumps({
                "quality_score": 85,
                "dimensions": {"structure": 90},
                "issues": [],
                "recommendations": []
            }))
        ]
        mock_llm.invoke.side_effect = mock_responses
        mock_chat_openai.return_value = mock_llm
        
        # Process the PDF
        processor = PDFProcessor(test_config)
        result = processor.process(sample_pdf_path, temp_dir)
        
        # Verify the result
        assert isinstance(result, dict)
        assert "status" in result
        # Note: Actual status depends on mocked responses
    
    def test_configuration_validation(self, temp_dir):
        """Test configuration validation and error handling."""
        # Test missing API key
        with pytest.raises(ValueError):
            Config(openai_api_key="")
        
        # Test invalid values with fallbacks
        config = Config(
            openai_api_key="test-key",
            temperature=-1,  # Invalid, should use default
            max_tokens=0,    # Invalid, should use default
            max_concurrent_pages=0  # Invalid, should use default
        )
        
        assert config.temperature == 0.1  # Default
        assert config.max_tokens == 4096  # Default
        assert config.max_concurrent_pages == 5  # Default
    
    def test_file_handling_edge_cases(self, test_config, temp_dir):
        """Test file handling edge cases."""
        processor = PDFProcessor(test_config)
        
        # Test nonexistent file
        result = processor.process("nonexistent.pdf", temp_dir)
        assert result["status"] == "failed"
        assert "error" in result
        
        # Test invalid output directory
        with pytest.raises((PermissionError, OSError)):
            processor.process("test.pdf", "/invalid/path/that/does/not/exist")
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_limits(self, test_config, temp_dir):
        """Test concurrent processing limits and resource management."""
        # Create config with limited concurrency
        limited_config = Config(
            openai_api_key="test-key",
            max_concurrent_pages=2  # Limited concurrency
        )
        
        # Create multiple PDF files
        pdf_files = []
        for i in range(5):
            pdf_path = temp_dir / f"concurrent_test{i}.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nSample content")
            pdf_files.append(str(pdf_path))
        
        with patch('pparser.processors.batch_processor.BatchWorkflow') as mock_workflow:
            mock_workflow_instance = Mock()
            mock_workflow_instance.process_batch = AsyncMock(return_value={
                "summary": {"total_files": 5, "completed": 5, "failed": 0},
                "results": [{"file": f, "status": "completed"} for f in pdf_files]
            })
            mock_workflow.return_value = mock_workflow_instance
            
            processor = BatchProcessor(limited_config)
            result = await processor.process_batch(pdf_files, temp_dir)
            
            # Should handle all files despite concurrency limits
            assert result["summary"]["total_files"] == 5
            assert result["summary"]["completed"] == 5
