# PPARSER System Documentation and Examples

## Overview
PPARSER is a complete multiagent system that converts digital PDFs to structured Markdown using LangChain/LangGraph and GPT-4o-mini. The system extracts text, images, tables, mathematical formulas, and forms while maintaining document structure.

## System Architecture

### Core Components

1. **Content Extractors** - Specialized extractors for different content types:
   - `TextExtractor`: Hierarchical text structure detection with PyMuPDF/pdfplumber
   - `ImageExtractor`: Image extraction, validation, and metadata generation
   - `TableExtractor`: Table detection, CSV export, and Markdown conversion
   - `FormulaExtractor`: Mathematical formula detection and LaTeX conversion
   - `FormExtractor`: Survey/questionnaire detection with interactive Markdown

2. **LLM Agents** - AI-powered analysis and enhancement:
   - `TextAnalysisAgent`: Structure analysis and content cleaning
   - `ImageAnalysisAgent`: Enhanced descriptions and positioning
   - `TableAnalysisAgent`: Content analysis and formatting improvement
   - `FormulaAnalysisAgent`: LaTeX conversion and mathematical classification
   - `FormAnalysisAgent`: Interactive form conversion and metadata
   - `StructureBuilderAgent`: Document assembly and Markdown generation
   - `QualityValidatorAgent`: Quality assessment and improvement suggestions

3. **LangGraph Workflows** - Orchestrated processing pipelines:
   - `PDFWorkflow`: Complete single-file processing pipeline with 11 workflow nodes
   - `BatchWorkflow`: Concurrent processing for multiple files with retry mechanism

4. **Main Processors** - High-level processing interfaces:
   - `PDFProcessor`: Enhanced single-file processing with quality validation
   - `BatchProcessor`: Comprehensive batch processing with detailed reporting

## Installation and Setup

### Requirements
- Python 3.8+
- OpenAI API key
- Required dependencies (see requirements.txt)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd PPARSER

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Set up environment variables
# Edit .env with your OpenAI API key
```

### Configuration
The system uses environment variables for configuration:
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: gpt-4o-mini)
- `OPENAI_TEMPERATURE`: Temperature setting (default: 0.1)
- `MAX_CONCURRENT_PAGES`: Max pages to process concurrently (default: 5)

## Usage Examples

### Command Line Interface

#### Process a Single PDF
```bash
python -m pparser single document.pdf -o output/
```

#### Batch Process Multiple PDFs
```bash
python -m pparser batch input_directory/ -o output_directory/
```

#### Process Specific Files from a List
```bash
python -m pparser filelist file_list.txt -o output_directory/
```

#### Advanced Options
```bash
# Disable quality validation
python -m pparser single document.pdf -o output/ --no-quality-check

# Custom number of workers for batch processing
python -m pparser batch input/ -o output/ --workers 8

# Recursive directory processing with pattern matching
python -m pparser batch input/ -o output/ --pattern "*.pdf" --recursive

# Custom configuration file
python -m pparser single document.pdf -o output/ --config custom_config.json
```

#### System Status and Workflow
```bash
# Check system status
python -m pparser status

# Display workflow visualization
python -m pparser workflow
```

### Python API Usage

#### Basic PDF Processing
```python
from pparser.processors import PDFProcessor
from pparser.config import Config

# Initialize with default configuration
config = Config()
processor = PDFProcessor(config)

# Process a single PDF
result = await processor.process_pdf("document.pdf", "output/")
print(f"Processing completed: {result.success}")
```

#### Batch Processing
```python
from pparser.processors import BatchProcessor
from pathlib import Path

# Initialize batch processor
processor = BatchProcessor(config)

# Process all PDFs in a directory
results = await processor.process_directory(
    input_dir=Path("input/"),
    output_dir=Path("output/"),
    max_workers=4
)

# Print summary
print(f"Processed: {results.total_files}")
print(f"Successful: {results.successful}")
print(f"Failed: {results.failed}")
```

#### Custom Workflow
```python
from pparser.workflows import PDFWorkflow

# Create custom workflow
workflow = PDFWorkflow()

# Define processing state
state = {
    "pdf_path": "document.pdf",
    "output_dir": "output/",
    "config": config
}

# Execute workflow
result = await workflow.workflow.ainvoke(state)
```

## Output Structure

The system generates structured output for each processed PDF:

```
output_directory/
├── document.md              # Main Markdown file
├── document_assets/         # Asset directory
│   ├── images/             # Extracted images
│   │   ├── image_001.png
│   │   └── image_002.jpg
│   ├── tables/             # Table data
│   │   ├── table_001.csv
│   │   └── table_002.md
│   └── metadata.json       # Processing metadata
└── document_report.json    # Quality assessment report
```

## Features

### Content Extraction
- **Text**: Hierarchical structure detection, font analysis, reading order
- **Images**: Format conversion, metadata extraction, positioning
- **Tables**: Multi-method detection, CSV export, Markdown formatting
- **Formulas**: LaTeX conversion, mathematical notation preservation
- **Forms**: Interactive element detection, survey reconstruction

### AI Enhancement
- **Structure Analysis**: Document hierarchy, section detection
- **Content Improvement**: Text cleaning, formatting enhancement
- **Context Understanding**: Semantic analysis, relationship detection
- **Quality Validation**: Completeness checking, accuracy assessment

### Processing Features
- **Concurrent Processing**: Multi-page and multi-file parallelization
- **Error Recovery**: Robust handling of corrupted or complex PDFs
- **Asset Management**: Organized file structure, proper linking
- **Progress Tracking**: Real-time status updates, detailed logging

## Testing

The system includes comprehensive testing:

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python -m pytest tests/test_extractors.py -v
python -m pytest tests/test_agents.py -v
python -m pytest tests/test_workflows.py -v

# Run with coverage
python -m pytest --cov=pparser --cov-report=html
```

## Performance Considerations

### Optimization Tips
1. **Batch Processing**: Use batch mode for multiple files
2. **Concurrent Pages**: Adjust `MAX_CONCURRENT_PAGES` based on system resources
3. **Model Selection**: Consider faster models for bulk processing
4. **Quality Checks**: Disable for faster processing when not needed

### Resource Usage
- **Memory**: ~1-2GB per concurrent PDF page
- **API Calls**: ~10-50 calls per PDF page (depending on content)
- **Processing Time**: ~30-120 seconds per page (depending on complexity)

## Troubleshooting

### Common Issues
1. **OpenAI API Key**: Ensure valid API key in environment
2. **Memory Errors**: Reduce concurrent pages or use smaller PDFs
3. **Rate Limits**: Implement delays or use lower-tier models
4. **PDF Corruption**: Check PDF integrity before processing

### Debug Mode
```bash
# Enable verbose logging
python -m pparser single document.pdf -o output/ --verbose

# Custom log file
python -m pparser single document.pdf -o output/ --log-file debug.log
```

## License
MIT License - see LICENSE file for details.
