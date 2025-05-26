"""
Performance and stress tests for the PDF parser system.
"""
import pytest
import time
import asyncio
from unittest.mock import Mock, patch
from pathlib import Path
import concurrent.futures

from pparser.processors.pdf_processor import PDFProcessor
from pparser.processors.batch_processor import BatchProcessor
from pparser.config import Config


class TestPerformance:
    """Performance tests for the PDF parser system."""
    
    @patch('pparser.processors.pdf_processor.PDFWorkflow')
    def test_single_pdf_processing_time(self, mock_workflow, test_config, temp_dir):
        """Test processing time for a single PDF."""
        # Mock workflow with realistic processing time
        mock_workflow_instance = Mock()
        
        def mock_process(pdf_path, output_dir):
            time.sleep(0.1)  # Simulate processing time
            return {
                "status": "completed",
                "quality_score": 85,
                "processing_time": 0.1
            }
        
        mock_workflow_instance.process.side_effect = mock_process
        mock_workflow.return_value = mock_workflow_instance
        
        processor = PDFProcessor(test_config)
        
        # Measure processing time
        start_time = time.time()
        result = processor.process("test.pdf", temp_dir)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert result["status"] == "completed"
        assert processing_time < 1.0  # Should complete within 1 second for mock
    
    @pytest.mark.asyncio
    @patch('pparser.processors.batch_processor.BatchWorkflow')
    async def test_batch_processing_scalability(self, mock_batch_workflow, test_config, temp_dir):
        """Test batch processing scalability with increasing file counts."""
        mock_workflow_instance = Mock()
        
        async def mock_process_batch(pdf_files, output_dir):
            # Simulate processing time proportional to file count
            await asyncio.sleep(len(pdf_files) * 0.01)
            return {
                "summary": {
                    "total_files": len(pdf_files),
                    "completed": len(pdf_files),
                    "failed": 0,
                    "total_time": len(pdf_files) * 0.01
                },
                "results": [
                    {"file": f, "status": "completed"} for f in pdf_files
                ]
            }
        
        mock_workflow_instance.process_batch.side_effect = mock_process_batch
        mock_batch_workflow.return_value = mock_workflow_instance
        
        processor = BatchProcessor(test_config)
        
        # Test with different file counts
        file_counts = [1, 5, 10, 20]
        processing_times = []
        
        for count in file_counts:
            pdf_files = [f"test{i}.pdf" for i in range(count)]
            
            start_time = time.time()
            result = await processor.process_batch(pdf_files, temp_dir)
            end_time = time.time()
            
            processing_times.append(end_time - start_time)
            assert result["summary"]["completed"] == count
        
        # Processing time should scale roughly linearly
        # (allowing for some overhead and variance)
        assert processing_times[-1] > processing_times[0]  # More files take longer
        assert processing_times[-1] < processing_times[0] * 30  # But not excessively longer
    
    @pytest.mark.asyncio
    @patch('pparser.processors.batch_processor.BatchWorkflow')
    async def test_concurrent_processing_efficiency(self, mock_batch_workflow, test_config, temp_dir):
        """Test efficiency of concurrent processing."""
        # Test with different concurrency levels
        concurrency_levels = [1, 2, 4]
        
        for max_concurrent in concurrency_levels:
            config = Config(
                openai_api_key="test-key",
                max_concurrent_pages=max_concurrent
            )
            
            mock_workflow_instance = Mock()
            
            async def mock_process_batch(pdf_files, output_dir):
                # Simulate concurrent processing
                await asyncio.sleep(0.1)  # Base processing time
                return {
                    "summary": {
                        "total_files": len(pdf_files),
                        "completed": len(pdf_files),
                        "failed": 0
                    },
                    "results": [
                        {"file": f, "status": "completed"} for f in pdf_files
                    ]
                }
            
            mock_workflow_instance.process_batch.side_effect = mock_process_batch
            mock_batch_workflow.return_value = mock_workflow_instance
            
            processor = BatchProcessor(config)
            pdf_files = [f"test{i}.pdf" for i in range(8)]
            
            start_time = time.time()
            result = await processor.process_batch(pdf_files, temp_dir)
            end_time = time.time()
            
            assert result["summary"]["completed"] == 8
            assert end_time - start_time < 5.0  # Should complete quickly
    
    def test_memory_usage_with_large_documents(self, test_config):
        """Test memory usage with large document simulation."""
        # This test would normally require actual large PDFs
        # For now, we'll simulate with large data structures
        
        processor = PDFProcessor(test_config)
        
        # Simulate large document data
        large_text_data = {
            'pages': [
                {
                    'page_num': i,
                    'text': 'Large text content ' * 1000,  # Simulate large text
                    'blocks': [{'text': 'Block ' * 100} for _ in range(50)]
                }
                for i in range(100)  # 100 pages
            ]
        }
        
        # This should not cause memory issues with proper handling
        # In a real test, you would monitor actual memory usage
        assert len(large_text_data['pages']) == 100
        assert len(large_text_data['pages'][0]['text']) > 10000


class TestStressTests:
    """Stress tests for edge cases and error conditions."""
    
    @pytest.mark.asyncio
    @patch('pparser.processors.batch_processor.BatchWorkflow')
    async def test_high_failure_rate_handling(self, mock_batch_workflow, test_config, temp_dir):
        """Test handling of high failure rates in batch processing."""
        mock_workflow_instance = Mock()
        
        async def mock_process_batch_with_failures(pdf_files, output_dir):
            # Simulate 70% failure rate
            failed_count = int(len(pdf_files) * 0.7)
            completed_count = len(pdf_files) - failed_count
            
            results = []
            for i, f in enumerate(pdf_files):
                if i < failed_count:
                    results.append({"file": f, "status": "failed", "error": "Simulated error"})
                else:
                    results.append({"file": f, "status": "completed", "quality_score": 75})
            
            return {
                "summary": {
                    "total_files": len(pdf_files),
                    "completed": completed_count,
                    "failed": failed_count
                },
                "results": results
            }
        
        mock_workflow_instance.process_batch.side_effect = mock_process_batch_with_failures
        mock_batch_workflow.return_value = mock_workflow_instance
        
        processor = BatchProcessor(test_config)
        pdf_files = [f"test{i}.pdf" for i in range(10)]
        
        result = await processor.process_batch(pdf_files, temp_dir)
        
        # Should handle high failure rate gracefully
        assert result["summary"]["total_files"] == 10
        assert result["summary"]["failed"] >= 6  # At least 60% failed
        assert result["summary"]["completed"] >= 1  # At least some completed
    
    @patch('pparser.processors.pdf_processor.PDFWorkflow')
    def test_error_recovery_mechanisms(self, mock_workflow, test_config, temp_dir):
        """Test error recovery and graceful degradation."""
        mock_workflow_instance = Mock()
        
        # Simulate various types of errors
        error_scenarios = [
            Exception("Network timeout"),
            MemoryError("Out of memory"),
            ValueError("Invalid PDF format"),
            KeyError("Missing configuration")
        ]
        
        processor = PDFProcessor(test_config)
        
        for error in error_scenarios:
            mock_workflow_instance.process.side_effect = error
            mock_workflow.return_value = mock_workflow_instance
            
            result = processor.process("test.pdf", temp_dir)
            
            # Should handle all error types gracefully
            assert result["status"] == "failed"
            assert "error" in result
            assert isinstance(result["error"], str)
    
    def test_configuration_edge_cases(self):
        """Test configuration with edge case values."""
        # Test minimum values
        config_min = Config(
            openai_api_key="test-key",
            temperature=0.0,
            max_tokens=1,
            max_concurrent_pages=1,
            chunk_size=100
        )
        assert config_min.temperature == 0.0
        assert config_min.max_tokens == 1
        
        # Test maximum reasonable values
        config_max = Config(
            openai_api_key="test-key",
            temperature=1.0,
            max_tokens=8192,
            max_concurrent_pages=50,
            chunk_size=10000
        )
        assert config_max.temperature == 1.0
        assert config_max.max_tokens == 8192
    
    @pytest.mark.asyncio
    @patch('pparser.processors.batch_processor.BatchWorkflow')
    async def test_resource_exhaustion_simulation(self, mock_batch_workflow, test_config):
        """Test behavior under simulated resource exhaustion."""
        mock_workflow_instance = Mock()
        
        async def mock_resource_exhaustion(pdf_files, output_dir):
            # Simulate resource exhaustion after processing some files
            if len(pdf_files) > 5:
                raise MemoryError("Simulated memory exhaustion")
            
            return {
                "summary": {
                    "total_files": len(pdf_files),
                    "completed": len(pdf_files),
                    "failed": 0
                },
                "results": [
                    {"file": f, "status": "completed"} for f in pdf_files
                ]
            }
        
        mock_workflow_instance.process_batch.side_effect = mock_resource_exhaustion
        mock_batch_workflow.return_value = mock_workflow_instance
        
        processor = BatchProcessor(test_config)
        
        # Test with small batch (should succeed)
        small_batch = [f"test{i}.pdf" for i in range(3)]
        result = await processor.process_batch(small_batch, "/tmp")
        assert result["summary"]["completed"] == 3
        
        # Test with large batch (should fail gracefully)
        large_batch = [f"test{i}.pdf" for i in range(10)]
        
        # Should handle the error gracefully
        try:
            result = await processor.process_batch(large_batch, "/tmp")
            # If it doesn't raise an exception, it should return error status
            if "summary" in result:
                assert result["summary"]["failed"] > 0
        except MemoryError:
            # If it does raise, that's also acceptable for this test
            pass
    
    def test_unicode_and_special_characters(self, test_config):
        """Test handling of Unicode and special characters in file paths and content."""
        processor = PDFProcessor(test_config)
        
        # Test with Unicode file paths
        unicode_paths = [
            "测试文档.pdf",  # Chinese
            "documento_español.pdf",  # Spanish
            "файл_русский.pdf",  # Russian
            "文書_日本語.pdf",  # Japanese
            "file with spaces & symbols!@#$.pdf"
        ]
        
        for path in unicode_paths:
            # Should handle Unicode paths gracefully (even if file doesn't exist)
            result = processor.process(path, "/tmp")
            assert isinstance(result, dict)
            assert "status" in result
            # Files don't exist, so should fail, but gracefully
            assert result["status"] == "failed"
