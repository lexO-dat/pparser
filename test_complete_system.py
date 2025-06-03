#!/usr/bin/env python3
"""
Complete system test to validate end-to-end PDF processing
"""

# use for test (its failing xd) cd /home/lexo/dev/PPARSER && python test_complete_system.py

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pparser import PDFProcessor, Config


"""Test complete PDF processing pipeline"""
async def test_complete_processing():
    
    # Setup paths
    input_pdf = project_root / "examples" / "test_data" / "input" / "test_document.pdf"
    output_dir = project_root / "examples" / "output"
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input PDF exists
    if not input_pdf.exists():
        print(f"Test PDF not found at: {input_pdf}")
        return False
    
    print(f"Testing with PDF: {input_pdf}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Initialize configuration
        config = Config.from_env()
        print(f"Configuration loaded successfully")
        
        # Initialize processor
        processor = PDFProcessor(config)
        print(f"PDFProcessor initialized successfully")
        
        # Process the PDF
        print(f"Starting PDF processing...")
        result = await processor.process(
            pdf_path=str(input_pdf),
            output_dir=str(output_dir)
        )
        
        if result and result.get('status') in ['success', 'acceptable_quality', 'low_quality']:
            print(f"PDF processing completed successfully!")
            output_files = result.get('output_files', {})
            markdown_file = output_files.get('markdown')
            print(f"Output file: {markdown_file}")
            
            # Check if output file exists
            if markdown_file and Path(markdown_file).exists():
                file_size = Path(markdown_file).stat().st_size
                print(f"Output file size: {file_size} bytes")
                
                # Show first few lines of the output
                try:
                    with open(markdown_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        print(f"First 10 lines of output:")
                        for i, line in enumerate(lines[:10], 1):
                            print(f"   {i:2d}: {line}")
                        
                        if len(lines) > 10:
                            print(f"   ... and {len(lines) - 10} more lines")
                
                except Exception as e:
                    print(f"Could not read output file: {e}")
                
                return True
            else:
                print(f"Output file not found: {markdown_file}")
                return False
        else:
            print(f"PDF processing failed: {result}")
            return False
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    print("=" * 60)
    print("PPARSER Complete System Test")
    print("=" * 60)
    
    success = await test_complete_processing()
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED! System is working correctly.")
    else:
        print("TESTS FAILED! Please check the errors above.")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
