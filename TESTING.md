# Testing Documentation

This document provides comprehensive information about the testing suite for the PDF Parser multiagent system.

## Overview

The testing suite includes:
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions and end-to-end workflows
- **Performance tests**: Test system performance and scalability
- **CLI tests**: Test command-line interface functionality
- **Stress tests**: Test error handling and edge cases

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and configuration
├── test_utils.py            # Utility function tests
├── test_config.py           # Configuration management tests
├── test_base_classes.py     # Abstract base class tests
├── test_extractors.py       # Content extractor tests
├── test_agents.py           # LLM agent tests
├── test_workflows.py        # Workflow orchestration tests
├── test_integration.py      # End-to-end integration tests
├── test_cli.py              # CLI interface tests
└── test_performance.py      # Performance and stress tests
```

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type performance
python run_tests.py --type cli

# Run with coverage
python run_tests.py --coverage

# Run in parallel
python run_tests.py --parallel

# Run fast tests only (skip slow tests)
python run_tests.py --type fast
```

### Using pytest directly

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run specific test files
pytest tests/test_utils.py
pytest tests/test_agents.py

# Run with coverage
pytest --cov=pparser --cov-report=html

# Run specific test markers
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run specific test classes or functions
pytest tests/test_config.py::TestConfig::test_config_default_values
pytest -k "test_text_extraction"
```

## Test Categories

### Unit Tests
Test individual components in isolation with mocked dependencies.

**Coverage includes:**
- Configuration management
- Utility functions
- Base classes
- Individual extractors
- Individual agents
- Workflow nodes

**Example:**
```python
def test_clean_text_removes_extra_whitespace():
    text = "  Hello    world  \n\n  Test  "
    result = clean_text(text)
    assert result == "Hello world Test"
```

### Integration Tests
Test component interactions and end-to-end workflows.

**Coverage includes:**
- Complete PDF processing pipeline
- Agent communication
- Workflow orchestration
- File I/O operations
- Error propagation

**Example:**
```python
@patch('pparser.extractors.text.fitz.open')
@patch('pparser.agents.base.ChatOpenAI')
def test_complete_pdf_processing_pipeline(mock_chat_openai, mock_fitz_open, test_config):
    # Test complete processing from PDF to Markdown
    # Mock PDF document, LLM responses, and verify end result
```

### Performance Tests
Test system performance, scalability, and resource usage.

**Coverage includes:**
- Processing time measurements
- Memory usage monitoring
- Concurrent processing efficiency
- Scalability with increasing file counts

**Example:**
```python
async def test_batch_processing_scalability(self, mock_batch_workflow, test_config):
    # Test with different file counts: 1, 5, 10, 20
    # Verify processing time scales appropriately
```

### CLI Tests
Test command-line interface functionality.

**Coverage includes:**
- Command parsing
- Option handling
- Output formatting
- Error reporting
- Environment variable usage

**Example:**
```python
def test_process_single_command(self, mock_processor, temp_dir):
    runner = CliRunner()
    result = runner.invoke(process_single, [str(pdf_path), '--output-dir', str(temp_dir)])
    assert result.exit_code == 0
```

### Stress Tests
Test error handling, edge cases, and system limits.

**Coverage includes:**
- High failure rates
- Resource exhaustion
- Invalid inputs
- Unicode handling
- Configuration edge cases

## Test Fixtures

### Configuration Fixtures
- `test_config`: Standard test configuration
- `temp_dir`: Temporary directory for test files

### Sample Data Fixtures
- `sample_pdf_path`: Mock PDF file
- `sample_text_content`: Sample extracted text data
- `sample_image_data`: Sample image extraction data
- `sample_table_data`: Sample table data
- `sample_formula_data`: Sample mathematical formula data
- `sample_form_data`: Sample form field data

### Mock Fixtures
- `mock_openai_response`: Mock OpenAI API response
- `mock_pdf_document`: Mock PDF document for testing
- `sample_image`: Test image object

## Mocking Strategy

### External Dependencies
All external dependencies are mocked to ensure:
- Tests run without API keys or network access
- Consistent and predictable test results
- Fast test execution
- No external service dependencies

**Mocked components:**
- OpenAI API calls (`ChatOpenAI`)
- PDF processing libraries (`fitz.open`, `pdfplumber`)
- File system operations (when needed)
- Network requests

### Mock Patterns
```python
# Mock OpenAI responses
@patch('pparser.agents.base.ChatOpenAI')
def test_agent_functionality(mock_chat_openai):
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = json.dumps({"result": "success"})
    mock_llm.invoke.return_value = mock_response
    mock_chat_openai.return_value = mock_llm

# Mock PDF documents
@patch('pparser.extractors.text.fitz.open')
def test_pdf_extraction(mock_fitz_open):
    mock_doc = Mock()
    mock_page = Mock()
    mock_page.get_text.return_value = "Sample text"
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz_open.return_value = mock_doc
```

## Coverage Goals

### Target Coverage Levels
- **Overall coverage**: ≥85%
- **Critical components**: ≥90%
- **Utility functions**: ≥95%
- **Error handling**: ≥80%

### Coverage Reporting
```bash
# Generate HTML coverage report
pytest --cov=pparser --cov-report=html

# View coverage in terminal
pytest --cov=pparser --cov-report=term

# Generate XML coverage for CI
pytest --cov=pparser --cov-report=xml
```

## Continuous Integration

### GitHub Actions Setup
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest --cov=pparser --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Best Practices

### Test Writing Guidelines
1. **One assertion per test**: Focus on testing one specific behavior
2. **Descriptive test names**: Clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Mock external dependencies**: Ensure tests are isolated
5. **Test edge cases**: Include boundary conditions and error cases

### Test Organization
1. **Group related tests**: Use test classes to organize related functionality
2. **Use meaningful fixtures**: Create reusable test data and configurations
3. **Document complex tests**: Add docstrings for non-obvious test logic
4. **Keep tests fast**: Use mocks to avoid slow operations

### Error Testing
```python
def test_error_handling():
    # Test that appropriate errors are raised
    with pytest.raises(ValueError, match="Invalid input"):
        function_under_test(invalid_input)

    # Test graceful error handling
    result = function_with_error_handling(problematic_input)
    assert result["status"] == "failed"
    assert "error" in result
```

## Debugging Tests

### Running Individual Tests
```bash
# Run a specific test file
pytest tests/test_utils.py -v

# Run a specific test class
pytest tests/test_config.py::TestConfig -v

# Run a specific test method
pytest tests/test_config.py::TestConfig::test_config_default_values -v

# Run tests matching a pattern
pytest -k "test_text" -v
```

### Debug Output
```bash
# Show print statements
pytest -s

# Show full traceback
pytest --tb=long

# Drop into debugger on failure
pytest --pdb

# Show local variables in traceback
pytest --tb=auto --showlocals
```

## Test Data Management

### Sample Files
Test sample files should be:
- Small and minimal for fast tests
- Representative of real-world inputs
- Stored in `tests/data/` directory (if needed)
- Created dynamically in tests when possible

### Cleanup
```python
@pytest.fixture
def temp_dir():
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Or add the project root to PYTHONPATH
export PYTHONPATH=/path/to/PPARSER:$PYTHONPATH
```

#### Async Test Issues
```python
# Use pytest-asyncio for async tests
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

#### Mock Issues
```python
# Ensure mocks are applied before imports
@patch('module.external_dependency')
def test_function(mock_dependency):
    # Mock must be configured before using the tested function
    mock_dependency.return_value = expected_value
```

### Environment Setup
```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows
pip install -r requirements.txt

# Set test environment variables
export OPENAI_API_KEY=test-key-for-mocking
export PYTHONPATH=/path/to/PPARSER:$PYTHONPATH
```
