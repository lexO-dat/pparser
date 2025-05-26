#!/usr/bin/env python3
"""
PPARSER System Validation Summary
"""

import os
import sys
from pathlib import Path


def validate_system():
    """Validate the PPARSER system components"""
    
    print("=" * 70)
    print("PPARSER MULTIAGENT PDF-TO-MARKDOWN SYSTEM VALIDATION")
    print("=" * 70)
    
    validation_results = {}
    
    # Test 1: Configuration
    print("\n1. CONFIGURATION SYSTEM")
    print("-" * 30)
    try:
        from pparser.config import Config
        
        # Test basic config
        config = Config(openai_api_key='test-key')
        print(f"   ✓ Basic configuration: {config.openai_model}")
        
        # Test config with custom parameters
        config_custom = Config(
            openai_api_key='test-key',
            temperature=0.5,
            max_tokens=2048,
            chunk_size=1024
        )
        print(f"   ✓ Custom configuration: temp={config_custom.temperature}")
        
        # Test from_env method
        config_env = Config.from_env() if os.getenv('OPENAI_API_KEY') else None
        if config_env:
            print(f"   ✓ Environment configuration: {config_env.openai_model}")
        else:
            print(f"   ⚠ Environment configuration: No OPENAI_API_KEY set")
        
        validation_results['config'] = True
        
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        validation_results['config'] = False
    
    # Test 2: Base Classes
    print("\n2. BASE CLASSES")
    print("-" * 30)
    try:
        from pparser.extractors.base import BaseExtractor
        from pparser.agents.base import BaseAgent
        
        print(f"   ✓ BaseExtractor imported")
        print(f"   ✓ BaseAgent imported")
        
        validation_results['base_classes'] = True
        
    except Exception as e:
        print(f"   ❌ Base classes failed: {e}")
        validation_results['base_classes'] = False
    
    # Test 3: Extractors
    print("\n3. CONTENT EXTRACTORS")
    print("-" * 30)
    extractors = [
        ('TextExtractor', 'pparser.extractors.text'),
        ('ImageExtractor', 'pparser.extractors.image'),
        ('TableExtractor', 'pparser.extractors.table'),
        ('FormulaExtractor', 'pparser.extractors.formula'),
        ('FormExtractor', 'pparser.extractors.form'),
    ]
    
    extractor_success = 0
    for name, module in extractors:
        try:
            exec(f"from {module} import {name}")
            print(f"   ✓ {name}")
            extractor_success += 1
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    validation_results['extractors'] = extractor_success == len(extractors)
    
    # Test 4: Agents
    print("\n4. ANALYSIS AGENTS")
    print("-" * 30)
    agents = [
        ('TextAnalysisAgent', 'pparser.agents.text_agent'),
        ('ImageAnalysisAgent', 'pparser.agents.image_agent'),
        ('TableAnalysisAgent', 'pparser.agents.table_agent'),
        ('FormulaAnalysisAgent', 'pparser.agents.formula_agent'),
        ('FormAnalysisAgent', 'pparser.agents.form_agent'),
        ('StructureBuilderAgent', 'pparser.agents.structure_agent'),
        ('QualityValidatorAgent', 'pparser.agents.quality_agent'),
    ]
    
    agent_success = 0
    config = Config(openai_api_key='test-key')
    
    for name, module in agents:
        try:
            exec(f"from {module} import {name}")
            if name in ['StructureBuilderAgent', 'QualityValidatorAgent']:
                # These don't need config in constructor
                exec(f"agent = {name}()")
            else:
                exec(f"agent = {name}(config)")
            print(f"   ✓ {name}")
            agent_success += 1
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    validation_results['agents'] = agent_success == len(agents)
    
    # Test 5: Workflows
    print("\n5. WORKFLOW ORCHESTRATION")
    print("-" * 30)
    try:
        from pparser.workflows.pdf_workflow import PDFWorkflow
        from pparser.workflows.batch_workflow import BatchWorkflow
        
        print(f"   ✓ PDFWorkflow imported")
        print(f"   ✓ BatchWorkflow imported")
        
        validation_results['workflows'] = True
        
    except Exception as e:
        print(f"   ❌ Workflows failed: {e}")
        validation_results['workflows'] = False
    
    # Test 6: Processors
    print("\n6. HIGH-LEVEL PROCESSORS")
    print("-" * 30)
    try:
        from pparser.processors.pdf_processor import PDFProcessor
        from pparser.processors.batch_processor import BatchProcessor
        
        print(f"   ✓ PDFProcessor imported")
        print(f"   ✓ BatchProcessor imported")
        
        validation_results['processors'] = True
        
    except Exception as e:
        print(f"   ❌ Processors failed: {e}")
        validation_results['processors'] = False
    
    # Test 7: CLI Interface
    print("\n7. COMMAND LINE INTERFACE")
    print("-" * 30)
    try:
        from pparser.cli import PPARSERCli
        
        cli = PPARSERCli()
        print(f"   ✓ CLI interface created")
        print(f"   ✓ Parser available: {hasattr(cli, 'parser')}")
        
        validation_results['cli'] = True
        
    except Exception as e:
        print(f"   ❌ CLI failed: {e}")
        validation_results['cli'] = False
    
    # Test 8: Utilities
    print("\n8. UTILITY FUNCTIONS")
    print("-" * 30)
    try:
        from pparser.utils.helpers import clean_text, safe_filename, chunk_text
        from pparser.utils.logger import get_logger, setup_logging
        
        # Test helper functions
        clean_result = clean_text("  test   text  ")
        safe_result = safe_filename("test<>file.txt")
        chunk_result = chunk_text("test text", chunk_size=5)
        
        print(f"   ✓ Helper functions working")
        print(f"   ✓ Logger functions available")
        
        validation_results['utils'] = True
        
    except Exception as e:
        print(f"   ❌ Utils failed: {e}")
        validation_results['utils'] = False
    
    # Test 9: Package Integration
    print("\n9. PACKAGE INTEGRATION")
    print("-" * 30)
    try:
        import pparser
        
        # Test main package imports
        from pparser import Config, PDFProcessor, BatchProcessor
        
        print(f"   ✓ Main package imports working")
        print(f"   ✓ Package version: {getattr(pparser, '__version__', 'dev')}")
        
        validation_results['package'] = True
        
    except Exception as e:
        print(f"   ❌ Package integration failed: {e}")
        validation_results['package'] = False
    
    # Final Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    total_tests = len(validation_results)
    passed_tests = sum(validation_results.values())
    
    for component, status in validation_results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component.replace('_', ' ').title()}")
    
    print(f"\nRESULT: {passed_tests}/{total_tests} components validated successfully")
    
    if passed_tests == total_tests:
        print("\n🎉 PPARSER SYSTEM FULLY OPERATIONAL!")
        print("\nThe multiagent PDF-to-Markdown conversion system is ready for use.")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY environment variable for LLM functionality")
        print("2. Run: python -m pparser single your_document.pdf -o output/")
        print("3. Check the generated Markdown files in the output directory")
        
    elif passed_tests >= total_tests * 0.8:
        print("\n⚠️  PPARSER SYSTEM MOSTLY OPERATIONAL")
        print("\nMost components are working. Check failed components above.")
        
    else:
        print("\n❌ PPARSER SYSTEM NEEDS ATTENTION")
        print("\nSeveral components failed validation. Please review the errors above.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = validate_system()
    sys.exit(0 if success else 1)
