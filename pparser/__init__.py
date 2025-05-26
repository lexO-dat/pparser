"""
PPARSER - Multiagent PDF to Markdown Converter

A sophisticated multiagent system for converting PDF documents to structured Markdown
using LangChain/LangGraph and specialized AI agents.

This package provides:
- Specialized extractors for text, images, tables, formulas, and forms
- AI-powered agents for content analysis and enhancement
- LangGraph workflows for coordinated processing
- Quality validation and structure building
- Batch processing capabilities
- Comprehensive CLI interface

Key Classes:
- PDFProcessor: Main processor for single files
- BatchProcessor: Batch processing for multiple files
- PDFWorkflow: LangGraph-based processing workflow
- Various specialized extractors and agents

Usage:
    from pparser import PDFProcessor
    
    processor = PDFProcessor()
    result = await processor.process("document.pdf", "output/")
"""

__version__ = "1.0.0"
__author__ = "PPARSER Team"
__description__ = "Multiagent PDF to Markdown converter with AI-powered content analysis"

# Main processing classes
from .processors import PDFProcessor, BatchProcessor
from .workflows import PDFWorkflow, BatchWorkflow
from .config import Config

# Extractors
from .extractors import (
    TextExtractor, ImageExtractor, TableExtractor,
    FormulaExtractor, FormExtractor
)

# Agents
from .agents import (
    TextAnalysisAgent, ImageAnalysisAgent, TableAnalysisAgent,
    FormulaAnalysisAgent, FormAnalysisAgent, StructureBuilderAgent,
    QualityValidatorAgent
)

__all__ = [
    # Main classes
    'PDFProcessor',
    'BatchProcessor', 
    'PDFWorkflow',
    'BatchWorkflow',
    'Config',
    
    # Extractors
    'TextExtractor',
    'ImageExtractor', 
    'TableExtractor',
    'FormulaExtractor',
    'FormExtractor',
    
    # Agents
    'TextAnalysisAgent',
    'ImageAnalysisAgent',
    'TableAnalysisAgent', 
    'FormulaAnalysisAgent',
    'FormAnalysisAgent',
    'StructureBuilderAgent',
    'QualityValidatorAgent'
]
