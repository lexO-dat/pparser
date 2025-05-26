"""
Unit tests for text extraction functionality.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import fitz  # PyMuPDF

from pparser.extractors.text import TextExtractor
from pparser.config import Config


class TestTextExtractor:
    """Test the TextExtractor class."""
    
    @patch('pparser.extractors.text.fitz.open')
    def test_text_extractor_initialization(self, mock_fitz_open, test_config):
        """Test TextExtractor initialization."""
        extractor = TextExtractor(test_config)
        assert extractor.config == test_config
    
    @patch('pparser.extractors.text.fitz.open')
    def test_extract_with_valid_pdf(self, mock_fitz_open, test_config, temp_dir, sample_pdf_path):
        """Test text extraction with valid PDF."""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.number = 0
        mock_page.get_text.return_value = "Sample text content"
        mock_page.rect = Mock(width=612, height=792)
        
        # Mock text blocks
        mock_page.get_text_dict.return_value = {
            'blocks': [
                {
                    'type': 0,  # Text block
                    'bbox': [50, 50, 200, 100],
                    'lines': [
                        {
                            'bbox': [50, 50, 200, 70],
                            'spans': [
                                {
                                    'text': 'Sample text',
                                    'bbox': [50, 50, 150, 70],
                                    'size': 12,
                                    'font': 'Arial'
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc
        
        extractor = TextExtractor(test_config)
        result = extractor.extract(sample_pdf_path, temp_dir)
        
        # Verify extraction results
        assert 'pages' in result
        assert len(result['pages']) == 1
        assert result['pages'][0]['page_num'] == 1
        assert 'text' in result['pages'][0]
        assert 'structure' in result
    
    @patch('pparser.extractors.text.fitz.open')
    def test_extract_with_multiple_pages(self, mock_fitz_open, test_config, temp_dir, sample_pdf_path):
        """Test text extraction with multiple pages."""
        # Mock PyMuPDF document with multiple pages
        mock_doc = Mock()
        mock_pages = []
        
        for i in range(3):
            mock_page = Mock()
            mock_page.number = i
            mock_page.get_text.return_value = f"Sample text from page {i + 1}"
            mock_page.rect = Mock(width=612, height=792)
            mock_page.get_text_dict.return_value = {'blocks': []}
            mock_pages.append(mock_page)
        
        mock_doc.__len__.return_value = 3
        mock_doc.__getitem__.side_effect = lambda i: mock_pages[i]
        mock_doc.__iter__.return_value = mock_pages
        mock_fitz_open.return_value = mock_doc
        
        extractor = TextExtractor(test_config)
        result = extractor.extract(sample_pdf_path, temp_dir)
        
        # Verify extraction results
        assert len(result['pages']) == 3
        for i, page in enumerate(result['pages']):
            assert page['page_num'] == i + 1
            assert f"page {i + 1}" in page['text']
    
    def test_analyze_text_structure(self, test_config):
        """Test text structure analysis."""
        extractor = TextExtractor(test_config)
        
        pages_data = [
            {
                'page_num': 1,
                'text': 'Title\n\nThis is the introduction section.\n\nSubheading\n\nThis is more content.',
                'blocks': []
            }
        ]
        
        structure = extractor._analyze_text_structure(pages_data)
        
        assert 'title' in structure
        assert 'sections' in structure
        assert isinstance(structure['sections'], list)
    
    def test_detect_headings(self, test_config):
        """Test heading detection."""
        extractor = TextExtractor(test_config)
        
        # Mock text blocks with different font sizes
        blocks = [
            {
                'text': 'Main Title',
                'bbox': [50, 50, 200, 80],
                'spans': [{'size': 18, 'font': 'Arial-Bold'}]
            },
            {
                'text': 'Regular text content',
                'bbox': [50, 100, 200, 120],
                'spans': [{'size': 12, 'font': 'Arial'}]
            },
            {
                'text': 'Subheading',
                'bbox': [50, 150, 200, 170],
                'spans': [{'size': 14, 'font': 'Arial-Bold'}]
            }
        ]
        
        headings = extractor._detect_headings(blocks)
        
        # Should detect headings based on font size and formatting
        assert len(headings) >= 1
        for heading in headings:
            assert 'text' in heading
            assert 'level' in heading
    
    def test_extract_text_hierarchy(self, test_config):
        """Test text hierarchy extraction."""
        extractor = TextExtractor(test_config)
        
        text_content = """
        Main Document Title
        
        Chapter 1: Introduction
        This is the introduction content.
        
        Section 1.1: Overview
        This is the overview content.
        
        Chapter 2: Methods
        This is the methods content.
        """
        
        hierarchy = extractor._extract_text_hierarchy(text_content)
        
        assert isinstance(hierarchy, list)
        # Should identify hierarchical structure
        if hierarchy:
            assert 'title' in hierarchy[0] or 'text' in hierarchy[0]
    
    @patch('pparser.extractors.text.fitz.open')
    def test_extract_with_extraction_error(self, mock_fitz_open, test_config, temp_dir, sample_pdf_path):
        """Test handling of extraction errors."""
        # Mock PyMuPDF to raise an exception
        mock_fitz_open.side_effect = Exception("Failed to open PDF")
        
        extractor = TextExtractor(test_config)
        result = extractor.extract(sample_pdf_path, temp_dir)
        
        # Should handle error gracefully
        assert result is not None
        assert 'pages' in result
        assert len(result['pages']) == 0
    
    def test_clean_extracted_text(self, test_config):
        """Test text cleaning functionality."""
        extractor = TextExtractor(test_config)
        
        dirty_text = "  Multiple   spaces\n\n\nExtra newlines\t\tTabs  "
        clean_text = extractor._clean_text(dirty_text)
        
        assert "Multiple spaces" in clean_text
        assert clean_text.count('\n') < dirty_text.count('\n')
        assert '\t' not in clean_text
    
    def test_extract_metadata(self, test_config):
        """Test metadata extraction."""
        extractor = TextExtractor(test_config)
        
        # Mock document metadata
        mock_metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'subject': 'Test Subject',
            'creator': 'Test Creator'
        }
        
        metadata = extractor._extract_metadata(mock_metadata)
        
        assert isinstance(metadata, dict)
        assert 'title' in metadata
        assert metadata['title'] == 'Test Document'
