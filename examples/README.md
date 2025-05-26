# PPARSER Examples

This directory contains practical examples demonstrating how to use the PPARSER system for various PDF processing scenarios.

## Available Examples

### 1. Basic Processing (`basic_processing.py`)
Demonstrates how to process a single PDF file with default settings.

**Usage:**
```bash
cd examples
python basic_processing.py
```

**Features shown:**
- Single PDF processing
- Basic configuration
- Result handling
- Output file organization

### 2. Batch Processing (`batch_processing.py`)
Shows how to process multiple PDF files in batch mode with concurrent processing.

**Usage:**
```bash
cd examples
python batch_processing.py
```

**Features shown:**
- Directory-based batch processing
- Concurrent file processing
- Progress tracking
- Error handling for failed files
- Batch statistics and reporting

### 3. Custom Workflow (`custom_workflow.py`)
Demonstrates how to create and execute custom workflows using LangGraph.

**Usage:**
```bash
cd examples
python custom_workflow.py
```

**Features shown:**
- Custom workflow creation
- Workflow state management
- Step-by-step processing
- Workflow visualization
- Advanced error handling

## Setup Requirements

Before running the examples, ensure you have:

1. **Installed PPARSER:**
   ```bash
   pip install -e .
   ```

2. **Configured environment:**
   - Copy `.env.example` to `.env`
   - Set your OpenAI API key in `.env`

3. **Sample PDF files:**
   - Place test PDF files in the appropriate directories
   - The examples will guide you on file placement

## Example Data Structure

When running the examples, the following directory structure will be created:

```
examples/
├── basic_processing.py
├── batch_processing.py
├── custom_workflow.py
├── README.md
├── sample.pdf              # Place your test PDF here
├── input_pdfs/             # Directory for batch processing
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── output/                 # Output from basic processing
├── batch_output/           # Output from batch processing
└── workflow_output/        # Output from custom workflow
```

## Common Usage Patterns

### Processing Academic Papers
```python
# Configure for academic content
config = Config()
config.extract_formulas = True
config.preserve_citations = True
config.quality_threshold = 0.9

processor = PDFProcessor(config)
result = await processor.process_pdf("paper.pdf", "output/")
```

### Processing Business Documents
```python
# Configure for business documents
config = Config()
config.extract_tables = True
config.extract_forms = True
config.format_headers = True

processor = PDFProcessor(config)
result = await processor.process_pdf("report.pdf", "output/")
```

### High-Volume Processing
```python
# Configure for bulk processing
config = Config()
config.max_concurrent_pages = 10
config.quality_check = False  # Faster processing
config.extract_images = False  # Skip if not needed

processor = BatchProcessor(config)
results = await processor.process_directory("bulk_pdfs/", "output/")
```

## Troubleshooting Examples

### Memory Issues
If you encounter memory issues with large PDFs:
```python
# Reduce concurrent processing
config.max_concurrent_pages = 2

# Process in smaller batches
processor = BatchProcessor(config)
results = await processor.process_directory(
    "pdfs/", "output/", 
    max_workers=1  # Single worker
)
```

### API Rate Limits
If you hit OpenAI API rate limits:
```python
# Add delays between requests
config.api_delay = 1.0  # 1 second delay

# Use smaller batch sizes
config.batch_size = 5
```

### Complex PDFs
For PDFs with complex layouts:
```python
# Enable advanced processing
config.advanced_layout_detection = True
config.multi_column_support = True
config.complex_table_detection = True
```

## Performance Tips

1. **Optimize for your use case:**
   - Disable unnecessary extractors
   - Adjust quality thresholds
   - Use appropriate batch sizes

2. **Monitor resource usage:**
   - Check memory consumption
   - Monitor API usage
   - Track processing times

3. **Scale appropriately:**
   - Use more workers for I/O bound tasks
   - Reduce workers for memory-intensive processing
   - Consider distributed processing for very large datasets

## Getting Help

If you encounter issues with the examples:

1. Check the main documentation in `USAGE.md`
2. Review the configuration in `pparser/config.py`
3. Enable verbose logging with `--verbose` flag
4. Check the test files for additional usage patterns