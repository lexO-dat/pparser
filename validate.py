#!/usr/bin/env python3
"""
Simple validation script for PPARSER system
"""

def test_imports():
    """Test all major imports."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        from pparser import __version__
        print(f"pparser version: {__version__}")
        
        from pparser.config import Config
        config = Config()
        print(f"Config loaded: {config.openai_model}")
        
        from pparser.utils.logger import get_logger
        logger = get_logger("test")
        print("Logger created")
        
        from pparser.utils.helpers import clean_text, safe_filename
        print("Helper functions loaded")
        
        # Test extractors
        from pparser.extractors import (
            TextExtractor, ImageExtractor, TableExtractor, 
            FormulaExtractor, FormExtractor
        )
        print("All extractors imported")
        
        # Test agents
        from pparser.agents import (
            TextAnalysisAgent, ImageAnalysisAgent, TableAnalysisAgent,
            FormulaAnalysisAgent, FormAnalysisAgent, StructureBuilderAgent,
            QualityValidatorAgent
        )
        print("All agents imported")
        
        # Test workflows
        from pparser.workflows import PDFWorkflow, BatchWorkflow
        print("Workflows imported")
        
        # Test processors
        from pparser.processors import PDFProcessor, BatchProcessor
        print("Processors imported")
        
        # Test CLI
        from pparser.cli import cli, process_single
        print("CLI imported")
        
        return True
        
    except Exception as e:
        print(f"Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        from pparser.utils.helpers import clean_text, safe_filename, generate_hash
        
        # Test text cleaning
        dirty_text = "  Hello   World!  \n\n  "
        clean = clean_text(dirty_text)
        assert clean == "Hello World!", f"Expected 'Hello World!', got '{clean}'"
        print("Text cleaning works")
        
        # Test filename safety
        unsafe_name = "file<>:name?.pdf"
        safe_name = safe_filename(unsafe_name)
        assert "<" not in safe_name and ">" not in safe_name
        print("Filename safety works")
        
        # Test hash generation
        test_content = "test content"
        hash_val = generate_hash(test_content)
        assert len(hash_val) == 8
        print("Hash generation works")
        
        return True
        
    except Exception as e:
        print(f"Functionality error: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from pparser.config import Config
        
        config = Config()
        
        # Check required fields
        assert config.openai_model, "OpenAI model not set"
        assert config.openai_temperature is not None, "Temperature not set"
        assert config.max_concurrent_pages > 0, "Concurrent pages must be positive"
        
        print(f"Configuration valid:")
        print(f"   Model: {config.openai_model}")
        print(f"   Temperature: {config.openai_temperature}")
        print(f"   Max pages: {config.max_concurrent_pages}")
        
        return True
        
    except Exception as e:
        print(f"Configuration error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("PPARSER System Validation")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"{test.__name__} failed")
        except Exception as e:
            print(f"{test.__name__} crashed: {e}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("All systems operational!")
        print("\nYou can now use PPARSER:")
        print("  python -m pparser --help")
        return 0
    else:
        print("Some systems failed validation")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
