# PPARSER CLI Usage Guide - Complete System Integration

## Overview

PPARSER now provides comprehensive PDF-to-Markdown conversion with both original and enhanced CLI interfaces, featuring the new enhanced architecture with improved agent management, error handling, and processing capabilities.

### Available Interfaces
1. **Original CLI** (`pparser.cli`) - Stable
2. **Enhanced CLI** (`pparser.enhanced_cli`) - Advanced interface with new architecture features

## Architecture Enhancements

The enhanced system includes:
- **AgentFactory**: Centralized agent creation and pipeline management
- **ConfigManager**: Advanced configuration handling
- **ErrorHandler**: Comprehensive error handling with retry logic
- **MemorySystem**: Advanced memory management for agents
- **ContentUtils**: Specialized content processing utilities
- **Enhanced Processors**: Improved single-file and batch processing

## Installation and Setup

First, ensure PPARSER is installed:
```bash
cd /home/lexo/dev/PPARSER
pip install -e .
```

## Standard CLI Usage

### Basic Commands

#### 1. Process a Single PDF File
```bash
# Basic processing
python -m pparser single input.pdf -o output/

# With verbose logging
python -m pparser single input.pdf -o output/ --verbose

# Without quality validation (faster)
python -m pparser single input.pdf -o output/ --no-quality-check

# Custom configuration
python -m pparser single input.pdf -o output/ --config config.json
```

#### 2. Batch Process Multiple PDFs
```bash
# Process all PDFs in a directory
python -m pparser batch input_directory/ -o output_directory/

# With custom workers and pattern
python -m pparser batch input_directory/ -o output_directory/ --workers 8 --pattern "*.pdf"

# Recursive processing with retry
python -m pparser batch input_directory/ -o output_directory/ --workers 4 --retry 2

# Skip subdirectories
python -m pparser batch input_directory/ -o output_directory/ --no-recursive
#### 4. System Status and Information
```bash
# Check system status
python -m pparser status

# Check workflow visualization
python -m pparser workflow
```

## Enhanced CLI Usage

The enhanced CLI provides advanced features and improved architecture:

### Key Enhancements
- **Pipeline Selection**: Choose optimized processing strategies
- **Enhanced Error Handling**: Automatic retry with intelligent backoff
- **Agent Management**: Monitor and control individual agents
- **Quality Validation**: Multi-dimensional quality scoring
- **Advanced Reporting**: Comprehensive processing statistics

### Enhanced CLI Commands

#### 1. Single File Processing with Pipelines
```bash
# Academic pipeline for research papers
python pparser/enhanced_cli.py single paper.pdf -o output/ --pipeline academic

# Technical pipeline for manuals
python pparser/enhanced_cli.py single manual.pdf -o output/ --pipeline technical

# Fast pipeline for quick conversion
python pparser/enhanced_cli.py single document.pdf -o output/ --pipeline fast

# With comprehensive validation
python pparser/enhanced_cli.py single file.pdf -o output/ --validate --enhance --agent-memory
```

#### 2. Advanced Batch Processing
```bash
# Batch with retry and quality validation
python pparser/enhanced_cli.py batch input/ -o output/ --workers 8 --retry 2

# Academic pipeline for research papers
python pparser/enhanced_cli.py batch papers/ -o output/ --pipeline academic --validate

# High-performance processing
python pparser/enhanced_cli.py batch input/ -o output/ --workers 16 --pipeline fast --no-quality-check
```

#### 3. System Management and Monitoring
```bash
# System status with detailed information
python pparser/enhanced_cli.py status --detailed

# Agent management
python pparser/enhanced_cli.py agents list
python pparser/enhanced_cli.py agents test
python pparser/enhanced_cli.py agents inspect formula

# Configuration management
python pparser/enhanced_cli.py configure show
python pparser/enhanced_cli.py configure set temperature 0.5
```

### Pipeline Types

1. **Standard Pipeline** (`--pipeline standard`)
   - Balanced speed and quality
   - Good for general documents
   - Default option

2. **Academic Pipeline** (`--pipeline academic`)
   - Optimized for research papers
   - Enhanced formula and table processing
   - Higher quality validation

3. **Technical Pipeline** (`--pipeline technical`)
   - Optimized for technical documentation
   - Enhanced structure detection
   - Comprehensive asset management

4. **Fast Pipeline** (`--pipeline fast`)
   - Prioritizes speed over quality
   - Reduced validation steps
   - Quick turnaround

### Standard CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--verbose, -v` | Enable verbose logging | False |
| `--config` | Path to custom config file | None |
| `--log-file` | Log file path | Console only |
| `--no-quality-check` | Disable quality validation | False |
| `--no-metadata` | Don't include metadata | False |
| `--workers` | Number of concurrent workers | 4 |
| `--pattern` | File pattern to match | "*.pdf" |
| `--retry` | Number of retry attempts | 0 |

## Enhanced CLI Usage

The enhanced CLI provides advanced features and better architecture:

### Basic Commands

#### 1. Single File Processing with Pipelines
```bash
# Standard pipeline
python -m pparser.enhanced_cli single input.pdf -o output/

# Academic pipeline (optimized for research papers)
python -m pparser.enhanced_cli single research_paper.pdf -o output/ --pipeline academic

# Technical pipeline (optimized for technical documents)
python -m pparser.enhanced_cli single manual.pdf -o output/ --pipeline technical

# Fast pipeline (speed over quality)
python -m pparser.enhanced_cli single document.pdf -o output/ --pipeline fast
```

#### 2. Enhanced Batch Processing
```bash
# Basic batch with enhanced features
python -m pparser.enhanced_cli batch input/ -o output/

# With comprehensive validation and enhancement
python -m pparser.enhanced_cli batch input/ -o output/ --validate --enhance

# Academic pipeline for research papers
python -m pparser.enhanced_cli batch papers/ -o output/ --pipeline academic --workers 6
```

#### 3. System Management
```bash
# Check system status
python -m pparser.enhanced_cli status

# Detailed system information
python -m pparser.enhanced_cli status --detailed

# Test agent pipeline
python -m pparser.enhanced_cli test-pipeline --pipeline academic

# Configuration management
python -m pparser.enhanced_cli config show
python -m pparser.enhanced_cli config set --key max_workers --value 8
```

### Enhanced CLI Features

| Feature | Description |
|---------|-------------|
| **Pipeline Selection** | Choose optimized processing pipelines |
| **Quality Validation** | Enhanced content validation and scoring |
| **Retry Logic** | Intelligent retry with exponential backoff |
| **Agent Management** | Monitor and control individual agents |
| **Configuration Management** | Dynamic configuration updates |
| **Performance Metrics** | Detailed processing statistics |
| **Error Recovery** | Advanced error handling and recovery |

### Pipeline Types

1. **Standard Pipeline** (`--pipeline standard`)
   - Balanced speed and quality
   - Good for general documents
   - Default option

2. **Academic Pipeline** (`--pipeline academic`)
   - Optimized for research papers
   - Enhanced formula and citation handling
   - Better table and figure processing

3. **Technical Pipeline** (`--pipeline technical`)
   - Optimized for technical documentation
   - Enhanced code block detection
   - Better diagram and flowchart handling

4. **Fast Pipeline** (`--pipeline fast`)
   - Prioritizes speed over quality
   - Basic validation only
   - Good for bulk processing

## Examples

### Example 1: Process Research Papers
```bash
# Process academic papers with enhanced quality
python -m pparser.enhanced_cli batch research_papers/ -o output/ \
  --pipeline academic \
  --validate \
  --enhance \
  --workers 4 \
  --retry 1
```

### Example 2: Quick Bulk Processing
```bash
# Fast processing of many documents
python -m pparser.enhanced_cli batch documents/ -o output/ \
  --pipeline fast \
  --workers 8 \
  --no-validation
```

### Example 3: High-Quality Single Document
```bash
# Maximum quality processing
python -m pparser.enhanced_cli single important_document.pdf -o output/ \
  --pipeline academic \
  --validate \
  --enhance \
  --verbose
```

## Output Structure

Both CLIs produce the same output structure:

```
output/
├── document_name.md           # Main markdown file
├── images/                    # Extracted images
│   ├── page_1_img_1.png
│   └── page_2_img_1.png
├── tables/                    # Extracted tables
│   ├── page_3_table_1.csv
│   └── page_4_table_1.csv
└── metadata.json             # Processing metadata
```

## Configuration Files

Create a `config.json` file for custom settings:

```json
{
  "max_workers": 6,
  "quality_threshold": 0.8,
  "enable_ocr": true,
  "output_format": "markdown",
  "image_extraction": {
    "enabled": true,
    "min_size": 100,
    "formats": ["png", "jpg"]
  },
  "table_extraction": {
    "enabled": true,
    "detect_headers": true,
    "output_format": "csv"
  }
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed
   pip install -e .
   
   # Check Python path
   python -c "import pparser; print(pparser.__file__)"
   ```

2. **Memory Issues**
   ```bash
   # Reduce workers
   python -m pparser batch input/ -o output/ --workers 2
   
   # Use fast pipeline
   python -m pparser.enhanced_cli batch input/ -o output/ --pipeline fast
   ```

3. **Quality Issues**
   ```bash
   # Use academic pipeline for better quality
   python -m pparser.enhanced_cli single document.pdf -o output/ --pipeline academic --validate
   ```

### Getting Help

```bash
# General help
python -m pparser --help

# Command-specific help
python -m pparser single --help
python -m pparser batch --help

# Enhanced CLI help
python -m pparser.enhanced_cli --help
python -m pparser.enhanced_cli single --help
```

## Performance Tips

1. **Adjust Workers**: Use 1-2 workers per CPU core
2. **Use Fast Pipeline**: For bulk processing where quality is less critical
3. **Disable Validation**: Skip `--validate` for faster processing
4. **Custom Configuration**: Tune settings for your specific use case
5. **Monitor Resources**: Use `--verbose` to track performance

## Next Steps

1. Try processing a sample PDF with both CLIs
2. Compare output quality between pipelines
3. Customize configuration for your needs
4. Set up batch processing workflows
5. Integrate with your existing tools
