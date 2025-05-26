#!/usr/bin/env python3
"""
Batch Processing Example

This example demonstrates how to process multiple PDF files in batch mode.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pparser.processors import BatchProcessor
from pparser.config import Config
from pparser.utils.logger import setup_logging


async def batch_process_pdfs():
    """Process multiple PDF files in batch mode."""
    
    # Set up logging
    setup_logging(level="INFO")
    
    # Load configuration
    config = Config()
    
    # Create batch processor
    processor = BatchProcessor(config)
    
    # Example directories (you can change these)
    input_dir = Path("input_pdfs")
    output_dir = Path("batch_output")
    
    print(f"🚀 Starting batch PDF processing...")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    try:
        # Check if input directory exists
        if not input_dir.exists():
            print(f"❌ Input directory not found: {input_dir}")
            print("Creating example directory structure...")
            input_dir.mkdir(exist_ok=True)
            print(f"📁 Created: {input_dir}")
            print("Please place PDF files in this directory and run the example again")
            return False
        
        # List PDF files
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {input_dir}")
            print("Please place PDF files in the input directory")
            return False
        
        print(f"📋 Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"   • {pdf_file.name}")
        print()
        
        # Process the files
        results = await processor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            max_workers=2,  # Adjust based on your system
            pattern="*.pdf",
            recursive=False
        )
        
        print(f"📊 Batch Processing Results:")
        print(f"   • Total files: {results.total_files}")
        print(f"   • Successful: {results.successful}")
        print(f"   • Failed: {results.failed}")
        print(f"   • Total time: {results.total_time:.2f} seconds")
        print(f"   • Average time per file: {results.average_time:.2f} seconds")
        print()
        
        if results.failed_files:
            print(f"❌ Failed files:")
            for failed_file, error in results.failed_files.items():
                print(f"   • {failed_file}: {error}")
            print()
        
        if results.successful > 0:
            print(f"✅ Batch processing completed!")
            print(f"📁 Output files are in: {output_dir}")
            return True
        else:
            print(f"❌ No files processed successfully")
            return False
            
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        return False


def main():
    """Main function."""
    print("🤖 PPARSER - Batch Processing Example")
    print("=" * 50)
    
    try:
        result = asyncio.run(batch_process_pdfs())
        if result:
            print("\n✨ Example completed successfully!")
        else:
            print("\n❌ Example failed")
            return 1
    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
