#!/usr/bin/env python3
"""
PPARSER System Demonstration Script
==================================

This script demonstrates the complete PPARSER multiagent system functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pparser import __version__
from pparser.config import Config
from pparser.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


async def demo_system_status():
    """Demonstrate system status and configuration."""
    print("🤖 PPARSER - Multiagent PDF to Markdown Converter")
    print("=" * 60)
    print(f"📦 Version: {__version__}")
    print()
    
    # Load configuration
    config = Config()
    print("⚙️  Configuration:")
    print(f"   • OpenAI Model: {config.openai_model}")
    print(f"   • Temperature: {config.openai_temperature}")
    print(f"   • Max Tokens: {config.openai_max_tokens}")
    print(f"   • Max Concurrent Pages: {config.max_concurrent_pages}")
    print(f"   • Chunk Size: {config.chunk_size}")
    print(f"   • Output Format: {config.output_format}")
    print()
    
    print("🔧 System Components:")
    
    # Test imports
    try:
        from pparser.extractors import (
            TextExtractor, ImageExtractor, TableExtractor, 
            FormulaExtractor, FormExtractor
        )
        print("   ✅ Content Extractors loaded successfully")
        
        from pparser.agents import (
            TextAnalysisAgent, ImageAnalysisAgent, TableAnalysisAgent,
            FormulaAnalysisAgent, FormAnalysisAgent, StructureBuilderAgent,
            QualityValidatorAgent
        )
        print("   ✅ LLM Agents loaded successfully")
        
        from pparser.workflows import PDFWorkflow, BatchWorkflow
        print("   ✅ LangGraph Workflows loaded successfully")
        
        from pparser.processors import PDFProcessor, BatchProcessor
        print("   ✅ Main Processors loaded successfully")
        
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    print()
    print("🚀 System Status: All components loaded successfully!")
    print()
    
    # Demonstrate workflow visualization
    try:
        from pparser.processors import PDFProcessor
        processor = PDFProcessor(config)
        
        print("📊 Workflow Visualization:")
        print("-" * 30)
        workflow_viz = processor.get_workflow_visualization()
        print("Mermaid diagram generated successfully")
        print(f"Diagram length: {len(workflow_viz)} characters")
        print()
        
    except Exception as e:
        print(f"   ⚠️  Workflow visualization error: {e}")
    
    print("🔍 Available CLI Commands:")
    print("   • python -m pparser single <pdf> -o <output>")
    print("   • python -m pparser batch <input_dir> -o <output_dir>")
    print("   • python -m pparser filelist <list.txt> -o <output_dir>")
    print("   • python -m pparser status")
    print("   • python -m pparser workflow")
    print()
    
    return True


async def demo_workflow_creation():
    """Demonstrate workflow creation without processing."""
    print("🔄 Workflow Architecture:")
    print("-" * 30)
    
    try:
        from pparser.workflows.pdf_workflow import PDFWorkflow
        from langgraph.graph import StateGraph
        
        # Create a workflow instance
        workflow = PDFWorkflow()
        
        print("   ✅ PDF Workflow created")
        print(f"   • Nodes: {len(workflow.workflow.nodes) if hasattr(workflow.workflow, 'nodes') else 'N/A'}")
        print("   • Processing stages: Extract → Analyze → Build → Validate")
        print()
        
        # Show workflow stages
        stages = [
            "1. Content Extraction (Text, Images, Tables, Formulas, Forms)",
            "2. LLM Analysis (Structure, Context, Enhancement)",
            "3. Document Assembly (Markdown Generation)",
            "4. Quality Validation (Content Check, Improvement)"
        ]
        
        for stage in stages:
            print(f"   {stage}")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Workflow creation error: {e}")


def main():
    """Main demonstration function."""
    setup_logging(level="INFO")
    
    print("\n" + "=" * 60)
    print("PPARSER SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Run the async demonstration
        result = asyncio.run(demo_system_status())
        
        if result:
            asyncio.run(demo_workflow_creation())
            
            print("✨ Demonstration completed successfully!")
            print()
            print("Next steps:")
            print("1. Prepare PDF files for processing")
            print("2. Use 'python -m pparser single <file>' to process individual PDFs")
            print("3. Use 'python -m pparser batch <dir>' for bulk processing")
            print("4. Check output Markdown files and extracted assets")
            print()
            
        else:
            print("❌ System validation failed")
            return 1
            
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
