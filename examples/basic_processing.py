#!/usr/bin/env python3
"""
Basic PDF Processing Example

This example demonstrates how to process a single PDF file using PPARSER.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pparser.processors import PDFProcessor
from pparser.config import Config
from pparser.utils.logger import setup_logging


async def process_single_pdf():
    """Process a single PDF file with basic settings."""
    
    # Set up logging
    setup_logging(level="INFO")
    
    # Load configuration
    config = Config()
    
    # Create processor
    processor = PDFProcessor(config)
    
    # Example PDF path (you can change this)
    pdf_path = "sample.pdf"
    output_dir = "output"
    
    print(f"Starting PDF processing...")
    print(f"Input: {pdf_path}")
    print(f"Output: {output_dir}")
    print()
    
    try:
        # Check if PDF exists
        if not Path(pdf_path).exists():
            print(f"PDF file not found: {pdf_path}")
            print("Please place a PDF file named 'sample.pdf' in the current directory")
            return False
        
        # Process the PDF
        result = await processor.process_pdf(pdf_path, output_dir)
        
        if result.success:
            print(f"Processing completed successfully!")
            print(f"Pages processed: {result.pages_processed}")
            print(f"Images extracted: {result.images_extracted}")
            print(f"Tables extracted: {result.tables_extracted}")
            print(f"Formulas extracted: {result.formulas_extracted}")
            print(f"Forms extracted: {result.forms_extracted}")
            print(f"Processing time: {result.processing_time:.2f} seconds")
            print()
            print(f"Output files:")
            output_path = Path(output_dir)
            if output_path.exists():
                for file in sorted(output_path.rglob("*")):
                    if file.is_file():
                        print(f"   {file}")
            
            return True
        else:
            print(f"Processing failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"Error during processing: {e}")
        return False


def main():
    """Main function."""
    print("PPARSER - Single PDF Processing Example")
    print("=" * 50)
    
    try:
        result = asyncio.run(process_single_pdf())
        if result:
            print("\nExample completed successfully!")
        else:
            print("\nExample failed")
            return 1
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 130
    except Exception as e:
        print(f"\nFatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
