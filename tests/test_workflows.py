"""
Unit tests for workflow orchestration.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from pathlib import Path

from pparser.workflows.pdf_workflow import PDFWorkflow
from pparser.workflows.batch_workflow import BatchWorkflow


class TestPDFWorkflow:
    """Test the PDFWorkflow class."""
    
    def test_pdf_workflow_initialization(self, test_config):
        """Test PDFWorkflow initialization."""
        workflow = PDFWorkflow(test_config)
        assert workflow.config == test_config
        assert workflow.graph is not None
    
    def test_workflow_state_initialization(self, test_config):
        """Test workflow state initialization."""
        workflow = PDFWorkflow(test_config)
        
        state = workflow._initialize_state("test.pdf", "/output")
        
        assert state["pdf_path"] == "test.pdf"
        assert state["output_dir"] == "/output"
        assert state["status"] == "initialized"
        assert "extracted_data" in state
        assert "analyzed_data" in state
        assert "assembled_document" in state
    
    @patch('pparser.workflows.pdf_workflow.TextExtractor')
    @patch('pparser.workflows.pdf_workflow.ImageExtractor')
    @patch('pparser.workflows.pdf_workflow.TableExtractor')
    @patch('pparser.workflows.pdf_workflow.FormulaExtractor')
    @patch('pparser.workflows.pdf_workflow.FormExtractor')
    def test_extraction_node(self, mock_form_ext, mock_formula_ext, mock_table_ext, 
                           mock_image_ext, mock_text_ext, test_config):
        """Test the extraction node."""
        # Mock extractors
        for mock_extractor in [mock_text_ext, mock_image_ext, mock_table_ext, 
                              mock_formula_ext, mock_form_ext]:
            mock_instance = Mock()
            mock_instance.extract.return_value = {"test": "data"}
            mock_extractor.return_value = mock_instance
        
        workflow = PDFWorkflow(test_config)
        
        # Test state
        state = {
            "pdf_path": "test.pdf",
            "output_dir": "/output",
            "extracted_data": {}
        }
        
        result = workflow._extract_content(state)
        
        assert "extracted_data" in result
        assert "text" in result["extracted_data"]
        assert "images" in result["extracted_data"]
        assert "tables" in result["extracted_data"]
        assert "formulas" in result["extracted_data"]
        assert "forms" in result["extracted_data"]
    
    @patch('pparser.workflows.pdf_workflow.TextAnalysisAgent')
    @patch('pparser.workflows.pdf_workflow.ImageAnalysisAgent')
    @patch('pparser.workflows.pdf_workflow.TableAnalysisAgent')
    @patch('pparser.workflows.pdf_workflow.FormulaAnalysisAgent')
    @patch('pparser.workflows.pdf_workflow.FormAnalysisAgent')
    def test_analysis_node(self, mock_form_agent, mock_formula_agent, mock_table_agent,
                          mock_image_agent, mock_text_agent, test_config):
        """Test the analysis node."""
        # Mock agents
        for mock_agent in [mock_text_agent, mock_image_agent, mock_table_agent,
                          mock_formula_agent, mock_form_agent]:
            mock_instance = Mock()
            mock_instance.analyze.return_value = {"enhanced": "data"}
            mock_agent.return_value = mock_instance
        
        workflow = PDFWorkflow(test_config)
        
        # Test state with extracted data
        state = {
            "extracted_data": {
                "text": {"pages": []},
                "images": {"images": []},
                "tables": {"tables": []},
                "formulas": {"formulas": []},
                "forms": {"forms": []}
            },
            "analyzed_data": {}
        }
        
        result = workflow._analyze_content(state)
        
        assert "analyzed_data" in result
        assert "text" in result["analyzed_data"]
        assert "images" in result["analyzed_data"]
        assert "tables" in result["analyzed_data"]
        assert "formulas" in result["analyzed_data"]
        assert "forms" in result["analyzed_data"]
    
    @patch('pparser.workflows.pdf_workflow.StructureBuilderAgent')
    def test_structure_building_node(self, mock_structure_agent, test_config):
        """Test the structure building node."""
        # Mock structure builder agent
        mock_instance = Mock()
        mock_instance.analyze.return_value = {
            "document_structure": {"title": "Test Doc"},
            "markdown_structure": {"toc": "# TOC"}
        }
        mock_structure_agent.return_value = mock_instance
        
        workflow = PDFWorkflow(test_config)
        
        # Test state with analyzed data
        state = {
            "analyzed_data": {
                "text": {"structure": {"title": "Test"}},
                "images": {"enhanced_descriptions": []},
                "tables": {"enhanced_tables": []},
                "formulas": {"enhanced_formulas": []},
                "forms": {"enhanced_forms": []}
            },
            "document_structure": {}
        }
        
        result = workflow._build_structure(state)
        
        assert "document_structure" in result
        assert "title" in result["document_structure"]
    
    @patch('pparser.workflows.pdf_workflow.QualityValidatorAgent')
    def test_quality_validation_node(self, mock_quality_agent, test_config):
        """Test the quality validation node."""
        # Mock quality validator agent
        mock_instance = Mock()
        mock_instance.analyze.return_value = {
            "quality_score": 85,
            "dimensions": {"structure": 90},
            "issues": [],
            "recommendations": []
        }
        mock_quality_agent.return_value = mock_instance
        
        workflow = PDFWorkflow(test_config)
        
        # Test state with assembled document
        state = {
            "assembled_document": {
                "markdown_content": "# Test\nContent",
                "assets": {"images": [], "tables": []}
            },
            "quality_report": {}
        }
        
        result = workflow._validate_quality(state)
        
        assert "quality_report" in result
        assert "quality_score" in result["quality_report"]
        assert result["quality_report"]["quality_score"] == 85
    
    def test_workflow_completion_check(self, test_config):
        """Test workflow completion logic."""
        workflow = PDFWorkflow(test_config)
        
        # Test incomplete state
        incomplete_state = {
            "status": "processing",
            "quality_report": {}
        }
        
        result = workflow._check_completion(incomplete_state)
        assert result["status"] != "completed"
        
        # Test complete state
        complete_state = {
            "status": "processing",
            "quality_report": {"quality_score": 85},
            "final_output": {"markdown_file": "test.md"}
        }
        
        result = workflow._check_completion(complete_state)
        assert result["status"] == "completed"
    
    @patch('pparser.workflows.pdf_workflow.TextExtractor')
    @patch('pparser.workflows.pdf_workflow.ImageExtractor')
    @patch('pparser.workflows.pdf_workflow.TableExtractor')
    @patch('pparser.workflows.pdf_workflow.FormulaExtractor')
    @patch('pparser.workflows.pdf_workflow.FormExtractor')
    @patch('pparser.workflows.pdf_workflow.TextAnalysisAgent')
    @patch('pparser.workflows.pdf_workflow.StructureBuilderAgent')
    @patch('pparser.workflows.pdf_workflow.QualityValidatorAgent')
    def test_full_workflow_execution(self, mock_quality, mock_structure, mock_text_agent,
                                   mock_form_ext, mock_formula_ext, mock_table_ext,
                                   mock_image_ext, mock_text_ext, test_config, temp_dir):
        """Test full workflow execution."""
        # Mock all extractors and agents
        for mock_extractor in [mock_text_ext, mock_image_ext, mock_table_ext,
                              mock_formula_ext, mock_form_ext]:
            mock_instance = Mock()
            mock_instance.extract.return_value = {"test": "data"}
            mock_extractor.return_value = mock_instance
        
        mock_text_agent_instance = Mock()
        mock_text_agent_instance.analyze.return_value = {"enhanced": "text"}
        mock_text_agent.return_value = mock_text_agent_instance
        
        mock_structure_instance = Mock()
        mock_structure_instance.analyze.return_value = {
            "document_structure": {"title": "Test"},
            "markdown_structure": {"content": "# Test"}
        }
        mock_structure.return_value = mock_structure_instance
        
        mock_quality_instance = Mock()
        mock_quality_instance.analyze.return_value = {
            "quality_score": 85,
            "dimensions": {},
            "issues": [],
            "recommendations": []
        }
        mock_quality.return_value = mock_quality_instance
        
        workflow = PDFWorkflow(test_config)
        
        # Execute workflow
        result = workflow.process("test.pdf", str(temp_dir))
        
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "completed"


class TestBatchWorkflow:
    """Test the BatchWorkflow class."""
    
    def test_batch_workflow_initialization(self, test_config):
        """Test BatchWorkflow initialization."""
        workflow = BatchWorkflow(test_config)
        assert workflow.config == test_config
    
    @pytest.mark.asyncio
    @patch('pparser.workflows.batch_workflow.PDFWorkflow')
    async def test_process_single_file(self, mock_pdf_workflow, test_config):
        """Test processing a single file in batch."""
        # Mock PDF workflow
        mock_workflow_instance = Mock()
        mock_workflow_instance.process.return_value = {
            "status": "completed",
            "quality_score": 85
        }
        mock_pdf_workflow.return_value = mock_workflow_instance
        
        workflow = BatchWorkflow(test_config)
        
        result = await workflow._process_single_file("test.pdf", "/output")
        
        assert isinstance(result, dict)
        assert "file" in result
        assert "status" in result
        assert result["file"] == "test.pdf"
    
    @pytest.mark.asyncio
    @patch('pparser.workflows.batch_workflow.PDFWorkflow')
    async def test_batch_processing(self, mock_pdf_workflow, test_config):
        """Test batch processing of multiple files."""
        # Mock PDF workflow
        mock_workflow_instance = Mock()
        mock_workflow_instance.process.return_value = {
            "status": "completed",
            "quality_score": 85
        }
        mock_pdf_workflow.return_value = mock_workflow_instance
        
        workflow = BatchWorkflow(test_config)
        
        pdf_files = ["test1.pdf", "test2.pdf", "test3.pdf"]
        results = await workflow.process_batch(pdf_files, "/output")
        
        assert isinstance(results, dict)
        assert "summary" in results
        assert "results" in results
        assert len(results["results"]) == 3
        
        # Check summary statistics
        summary = results["summary"]
        assert "total_files" in summary
        assert "completed" in summary
        assert summary["total_files"] == 3
    
    @pytest.mark.asyncio
    @patch('pparser.workflows.batch_workflow.PDFWorkflow')
    async def test_batch_processing_with_failures(self, mock_pdf_workflow, test_config):
        """Test batch processing with some failures."""
        # Mock PDF workflow with mixed results
        def mock_process(pdf_path, output_dir):
            if "fail" in pdf_path:
                raise Exception("Processing failed")
            return {"status": "completed", "quality_score": 85}
        
        mock_workflow_instance = Mock()
        mock_workflow_instance.process.side_effect = mock_process
        mock_pdf_workflow.return_value = mock_workflow_instance
        
        workflow = BatchWorkflow(test_config)
        
        pdf_files = ["test1.pdf", "fail.pdf", "test3.pdf"]
        results = await workflow.process_batch(pdf_files, "/output")
        
        summary = results["summary"]
        assert summary["total_files"] == 3
        assert summary["completed"] < 3  # Some files failed
        assert summary["failed"] > 0
    
    def test_progress_tracking(self, test_config):
        """Test progress tracking functionality."""
        workflow = BatchWorkflow(test_config)
        
        # Initialize progress
        workflow._init_progress(5)
        assert workflow.progress["total"] == 5
        assert workflow.progress["completed"] == 0
        
        # Update progress
        workflow._update_progress("test1.pdf", "completed")
        assert workflow.progress["completed"] == 1
        
        workflow._update_progress("test2.pdf", "failed")
        assert workflow.progress["failed"] == 1
        
        # Get progress
        progress = workflow.get_progress()
        assert progress["total"] == 5
        assert progress["completed"] == 1
        assert progress["failed"] == 1
        assert progress["percentage"] == 40.0  # (1 + 1) / 5 * 100
