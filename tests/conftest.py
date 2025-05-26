"""
Test configuration and fixtures for the PDF Parser test suite.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator
import pytest
from unittest.mock import Mock, patch
import io
from PIL import Image

from pparser.config import Config


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_pdf_path(temp_dir: Path) -> Path:
    """Create a sample PDF file for testing."""
    # In a real test, you would create or copy a sample PDF here
    # For now, we'll create a placeholder file
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
    return pdf_path


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample image for testing."""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = Mock()
    mock_response.content = "Test response from OpenAI"
    mock_response.additional_kwargs = {}
    return mock_response


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create a test configuration."""
    return Config(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        output_dir=temp_dir / "output",
        max_concurrent_pages=2,
        chunk_size=1024,
        log_level="DEBUG"
    )


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return {
        'pages': [
            {
                'page_num': 1,
                'text': 'This is a sample document with multiple paragraphs.',
                'bbox': [0, 0, 100, 100],
                'blocks': [
                    {
                        'text': 'This is a sample document',
                        'bbox': [0, 0, 50, 20],
                        'type': 'text'
                    }
                ]
            }
        ],
        'structure': {
            'title': 'Sample Document',
            'sections': ['Introduction', 'Content']
        }
    }


@pytest.fixture
def sample_table_data():
    """Sample table data for testing."""
    return {
        'tables': [
            {
                'page_num': 1,
                'bbox': [10, 10, 90, 90],
                'data': [
                    ['Header 1', 'Header 2'],
                    ['Row 1 Col 1', 'Row 1 Col 2'],
                    ['Row 2 Col 1', 'Row 2 Col 2']
                ],
                'csv_path': None
            }
        ]
    }


@pytest.fixture
def sample_image_data(temp_dir: Path):
    """Sample image data for testing."""
    # Create a test image file
    image_path = temp_dir / "test_image.png"
    img = Image.new('RGB', (100, 100), color='blue')
    img.save(image_path)
    
    return {
        'images': [
            {
                'page_num': 1,
                'bbox': [20, 20, 80, 80],
                'image_path': str(image_path),
                'format': 'PNG',
                'size': (100, 100)
            }
        ]
    }


@pytest.fixture
def sample_formula_data():
    """Sample formula data for testing."""
    return {
        'formulas': [
            {
                'page_num': 1,
                'bbox': [30, 30, 70, 50],
                'text': 'E = mc²',
                'latex': r'E = mc^2',
                'type': 'equation'
            }
        ]
    }


@pytest.fixture
def sample_form_data():
    """Sample form data for testing."""
    return {
        'forms': [
            {
                'page_num': 1,
                'fields': [
                    {
                        'type': 'text',
                        'label': 'Name',
                        'bbox': [10, 50, 90, 60],
                        'required': True
                    },
                    {
                        'type': 'checkbox',
                        'label': 'Subscribe to newsletter',
                        'bbox': [10, 70, 20, 80],
                        'required': False
                    }
                ]
            }
        ]
    }


class MockPDFDocument:
    """Mock PDF document for testing."""
    
    def __init__(self, page_count=1):
        self.page_count = page_count
        self._pages = [MockPDFPage(i) for i in range(page_count)]
    
    def __len__(self):
        return self.page_count
    
    def __getitem__(self, index):
        return self._pages[index]
    
    def close(self):
        pass


class MockPDFPage:
    """Mock PDF page for testing."""
    
    def __init__(self, page_num):
        self.number = page_num
        self.rect = MockRect(0, 0, 612, 792)  # Standard page size
    
    def get_text(self, option="text"):
        return f"Sample text from page {self.number + 1}"
    
    def get_images(self):
        return []
    
    def get_drawings(self):
        return []


class MockRect:
    """Mock rectangle for testing."""
    
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = x1 - x0
        self.height = y1 - y0


@pytest.fixture
def mock_pdf_document():
    """Mock PDF document fixture."""
    return MockPDFDocument(page_count=3)
