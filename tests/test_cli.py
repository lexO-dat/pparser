"""
Unit tests for CLI functionality.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner
from pathlib import Path
import json

from pparser.cli import cli, process_single, process_batch, process_filelist


class TestCLI:
    """Test the CLI interface."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'PDF Parser' in result.output
        assert 'process-single' in result.output
        assert 'process-batch' in result.output
        assert 'process-filelist' in result.output
    
    @patch('pparser.cli.PDFProcessor')
    def test_process_single_command(self, mock_processor, temp_dir):
        """Test process-single CLI command."""
        # Create a sample PDF file
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nSample content")
        
        # Mock processor
        mock_processor_instance = Mock()
        mock_processor_instance.process.return_value = {
            "status": "completed",
            "quality_score": 85,
            "final_output": {
                "markdown_file": str(temp_dir / "output.md")
            }
        }
        mock_processor.return_value = mock_processor_instance
        
        runner = CliRunner()
        result = runner.invoke(process_single, [
            str(pdf_path),
            '--output-dir', str(temp_dir),
            '--model', 'gpt-4o-mini'
        ])
        
        assert result.exit_code == 0
        assert "Processing completed successfully" in result.output
        assert "Quality score: 85" in result.output
    
    @patch('pparser.cli.PDFProcessor')
    def test_process_single_with_error(self, mock_processor, temp_dir):
        """Test process-single command with processing error."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nSample content")
        
        # Mock processor to return error
        mock_processor_instance = Mock()
        mock_processor_instance.process.return_value = {
            "status": "failed",
            "error": "Processing failed due to invalid PDF"
        }
        mock_processor.return_value = mock_processor_instance
        
        runner = CliRunner()
        result = runner.invoke(process_single, [
            str(pdf_path),
            '--output-dir', str(temp_dir)
        ])
        
        assert result.exit_code != 0
        assert "Processing failed" in result.output
    
    def test_process_single_invalid_file(self):
        """Test process-single with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(process_single, [
            'nonexistent.pdf',
            '--output-dir', '/tmp'
        ])
        
        assert result.exit_code != 0
        assert "does not exist" in result.output
    
    @patch('pparser.cli.BatchProcessor')
    def test_process_batch_command(self, mock_batch_processor, temp_dir):
        """Test process-batch CLI command."""
        # Create sample PDF files
        pdf_dir = temp_dir / "pdfs"
        pdf_dir.mkdir()
        
        for i in range(3):
            pdf_path = pdf_dir / f"test{i}.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nSample content")
        
        # Mock batch processor
        mock_processor_instance = Mock()
        mock_processor_instance.process_directory = AsyncMock(return_value={
            "summary": {
                "total_files": 3,
                "completed": 3,
                "failed": 0,
                "average_quality": 82.5
            },
            "results": [
                {"file": "test0.pdf", "status": "completed", "quality_score": 85},
                {"file": "test1.pdf", "status": "completed", "quality_score": 80},
                {"file": "test2.pdf", "status": "completed", "quality_score": 82}
            ]
        })
        mock_batch_processor.return_value = mock_processor_instance
        
        runner = CliRunner()
        result = runner.invoke(process_batch, [
            str(pdf_dir),
            '--output-dir', str(temp_dir),
            '--concurrent', '2'
        ])
        
        assert result.exit_code == 0
        assert "Batch processing completed successfully" in result.output
        assert "Total files: 3" in result.output
        assert "Completed: 3" in result.output
    
    @patch('pparser.cli.BatchProcessor')
    def test_process_batch_with_failures(self, mock_batch_processor, temp_dir):
        """Test process-batch command with some failures."""
        pdf_dir = temp_dir / "pdfs"
        pdf_dir.mkdir()
        
        # Mock batch processor with failures
        mock_processor_instance = Mock()
        mock_processor_instance.process_directory = AsyncMock(return_value={
            "summary": {
                "total_files": 3,
                "completed": 2,
                "failed": 1,
                "average_quality": 75.0
            },
            "results": [
                {"file": "test0.pdf", "status": "completed", "quality_score": 80},
                {"file": "test1.pdf", "status": "failed", "error": "Processing error"},
                {"file": "test2.pdf", "status": "completed", "quality_score": 70}
            ]
        })
        mock_batch_processor.return_value = mock_processor_instance
        
        runner = CliRunner()
        result = runner.invoke(process_batch, [
            str(pdf_dir),
            '--output-dir', str(temp_dir)
        ])
        
        assert result.exit_code == 0  # Should complete with warnings
        assert "Failed: 1" in result.output
        assert "Some files failed to process" in result.output
    
    @patch('pparser.cli.BatchProcessor')
    def test_process_filelist_command(self, mock_batch_processor, temp_dir):
        """Test process-filelist CLI command."""
        # Create PDF files and file list
        pdf_files = []
        for i in range(2):
            pdf_path = temp_dir / f"list_test{i}.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nSample content")
            pdf_files.append(str(pdf_path))
        
        file_list_path = temp_dir / "files.txt"
        file_list_path.write_text("\n".join(pdf_files))
        
        # Mock batch processor
        mock_processor_instance = Mock()
        mock_processor_instance.process_file_list = AsyncMock(return_value={
            "summary": {
                "total_files": 2,
                "completed": 2,
                "failed": 0,
                "average_quality": 85.0
            },
            "results": [
                {"file": pdf_files[0], "status": "completed", "quality_score": 85},
                {"file": pdf_files[1], "status": "completed", "quality_score": 85}
            ]
        })
        mock_batch_processor.return_value = mock_processor_instance
        
        runner = CliRunner()
        result = runner.invoke(process_filelist, [
            str(file_list_path),
            '--output-dir', str(temp_dir)
        ])
        
        assert result.exit_code == 0
        assert "File list processing completed successfully" in result.output
        assert "Total files: 2" in result.output
    
    def test_process_filelist_invalid_file(self):
        """Test process-filelist with non-existent file list."""
        runner = CliRunner()
        result = runner.invoke(process_filelist, [
            'nonexistent_list.txt',
            '--output-dir', '/tmp'
        ])
        
        assert result.exit_code != 0
        assert "does not exist" in result.output
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('pparser.cli.PDFProcessor')
    def test_cli_with_environment_variables(self, mock_processor, temp_dir):
        """Test CLI with environment variables."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nSample content")
        
        mock_processor_instance = Mock()
        mock_processor_instance.process.return_value = {
            "status": "completed",
            "quality_score": 85
        }
        mock_processor.return_value = mock_processor_instance
        
        runner = CliRunner()
        result = runner.invoke(process_single, [
            str(pdf_path),
            '--output-dir', str(temp_dir)
        ])
        
        assert result.exit_code == 0
        # Should use environment variable for API key
        mock_processor.assert_called_once()
    
    def test_cli_missing_api_key(self, temp_dir):
        """Test CLI without API key."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nSample content")
        
        with patch.dict('os.environ', {}, clear=True):
            runner = CliRunner()
            result = runner.invoke(process_single, [
                str(pdf_path),
                '--output-dir', str(temp_dir)
            ])
            
            assert result.exit_code != 0
            assert "API key" in result.output or "required" in result.output
    
    @patch('pparser.cli.PDFProcessor')
    def test_cli_with_custom_options(self, mock_processor, temp_dir):
        """Test CLI with custom processing options."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nSample content")
        
        mock_processor_instance = Mock()
        mock_processor_instance.process.return_value = {
            "status": "completed",
            "quality_score": 85
        }
        mock_processor.return_value = mock_processor_instance
        
        runner = CliRunner()
        result = runner.invoke(process_single, [
            str(pdf_path),
            '--output-dir', str(temp_dir),
            '--model', 'gpt-4',
            '--temperature', '0.3',
            '--max-tokens', '2048',
            '--chunk-size', '1024',
            '--log-level', 'DEBUG'
        ])
        
        assert result.exit_code == 0
        
        # Verify processor was called with custom config
        mock_processor.assert_called_once()
        config_arg = mock_processor.call_args[0][0]
        assert config_arg.openai_model == 'gpt-4'
        assert config_arg.temperature == 0.3
        assert config_arg.max_tokens == 2048
        assert config_arg.chunk_size == 1024
        assert config_arg.log_level == 'DEBUG'
    
    @patch('pparser.cli.PDFProcessor')
    def test_cli_output_formats(self, mock_processor, temp_dir):
        """Test CLI output format options."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nSample content")
        
        mock_processor_instance = Mock()
        mock_processor_instance.process.return_value = {
            "status": "completed",
            "quality_score": 85,
            "final_output": {
                "markdown_file": str(temp_dir / "output.md")
            }
        }
        mock_processor.return_value = mock_processor_instance
        
        runner = CliRunner()
        
        # Test quiet output
        result = runner.invoke(process_single, [
            str(pdf_path),
            '--output-dir', str(temp_dir),
            '--quiet'
        ])
        
        assert result.exit_code == 0
        assert result.output.strip() == ""  # Minimal output in quiet mode
        
        # Test verbose output
        result = runner.invoke(process_single, [
            str(pdf_path),
            '--output-dir', str(temp_dir),
            '--verbose'
        ])
        
        assert result.exit_code == 0
        assert len(result.output) > 100  # More detailed output
    
    @patch('pparser.cli.PDFProcessor')
    def test_cli_json_output(self, mock_processor, temp_dir):
        """Test CLI JSON output format."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nSample content")
        
        mock_processor_instance = Mock()
        mock_processor_instance.process.return_value = {
            "status": "completed",
            "quality_score": 85,
            "final_output": {
                "markdown_file": str(temp_dir / "output.md")
            },
            "metadata": {
                "pages": 5,
                "processing_time": 30.5
            }
        }
        mock_processor.return_value = mock_processor_instance
        
        runner = CliRunner()
        result = runner.invoke(process_single, [
            str(pdf_path),
            '--output-dir', str(temp_dir),
            '--output-format', 'json'
        ])
        
        assert result.exit_code == 0
        
        # Should output valid JSON
        try:
            output_data = json.loads(result.output)
            assert output_data["status"] == "completed"
            assert output_data["quality_score"] == 85
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
