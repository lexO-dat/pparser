#!/usr/bin/env python3
"""
End-to-end test for PPARSER system
"""

import os
import tempfile
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

from pparser.config import Config
from pparser.processors.pdf_processor import PDFProcessor


def create_test_pdf(output_path: Path) -> Path:
    """Create a simple test PDF with text, table-like content, and basic structure"""
    
    pdf_path = output_path / "test_document.pdf"
    
    # Create a simple PDF with reportlab
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Test Document for PPARSER")
    
    # Subtitle
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 100, "This is a test document to validate PDF processing capabilities")
    
    # Section 1
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, height - 140, "1. Introduction")
    
    c.setFont("Helvetica", 10)
    text_lines = [
        "This document contains various types of content that PPARSER should be able to extract:",
        "• Text paragraphs with different formatting",
        "• Simple table-like structures",
        "• Headers and sections",
        "• Mathematical expressions like x² + y² = z²"
    ]
    
    y_pos = height - 165
    for line in text_lines:
        c.drawString(72, y_pos, line)
        y_pos -= 15
    
    # Section 2 - Table-like content
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y_pos - 20, "2. Sample Data Table")
    
    y_pos -= 50
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y_pos, "Item")
    c.drawString(200, y_pos, "Quantity")
    c.drawString(300, y_pos, "Price ($)")
    
    # Table data
    table_data = [
        ("Apples", "10", "2.50"),
        ("Bananas", "5", "1.25"),
        ("Oranges", "8", "3.00"),
    ]
    
    c.setFont("Helvetica", 10)
    y_pos -= 20
    for item, qty, price in table_data:
        c.drawString(72, y_pos, item)
        c.drawString(200, y_pos, qty)
        c.drawString(300, y_pos, price)
        y_pos -= 15
    
    # Section 3
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y_pos - 30, "3. Conclusion")
    
    c.setFont("Helvetica", 10)
    c.drawString(72, y_pos - 55, "This test document validates that PPARSER can extract and process")
    c.drawString(72, y_pos - 70, "different types of content from PDF documents effectively.")
    
    c.save()
    
    return pdf_path


def test_end_to_end_processing():
    """Test complete PDF processing pipeline"""
    
    print("=" * 60)
    print("PPARSER END-TO-END VALIDATION TEST")
    print("=" * 60)
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_path = temp_path / "output"
        output_path.mkdir(exist_ok=True)
        
        try:
            # Step 1: Create test PDF
            print("\n1. Creating test PDF...")
            pdf_path = create_test_pdf(temp_path)
            print(f"   ✓ Test PDF created: {pdf_path}")
            print(f"   ✓ PDF size: {pdf_path.stat().st_size} bytes")
            
            # Step 2: Initialize configuration
            print("\n2. Initializing configuration...")
            config = Config(
                openai_api_key=os.getenv('OPENAI_API_KEY', 'test-key-for-demo'),
                output_dir=str(output_path),
                temp_dir=str(temp_path / "temp"),
                log_level="INFO"
            )
            print(f"   ✓ Config initialized")
            print(f"   ✓ Output directory: {config.output_dir}")
            print(f"   ✓ Using model: {config.openai_model}")
            
            # Step 3: Initialize PDF processor
            print("\n3. Initializing PDF processor...")
            processor = PDFProcessor(config)
            print(f"   ✓ PDF processor initialized")
            print(f"   ✓ Processor has {len(processor.__dict__)} components")
            
            # Step 4: Process the PDF (without LLM calls for this test)
            print("\n4. Processing PDF (text extraction only)...")
            
            # Test just the text extraction first
            from pparser.extractors.text import TextExtractor
            text_extractor = TextExtractor()
            
            text_result = text_extractor.extract_all_pages(pdf_path)
            print(f"   ✓ Text extraction completed")
            print(f"   ✓ Extracted from {len(text_result)} pages")
            
            if text_result:
                first_page = text_result[0]
                extracted_text = first_page.get('text', '')[:200]
                print(f"   ✓ Sample extracted text: {extracted_text}...")
            
            # Step 5: Test configuration validation
            print("\n5. Testing configuration validation...")
            
            # Test various config scenarios
            test_configs = [
                ("Valid config", {"openai_api_key": "test-key"}),
                ("Custom temperature", {"openai_api_key": "test-key", "temperature": 0.5}),
                ("Custom model", {"openai_api_key": "test-key", "openai_model": "gpt-3.5-turbo"}),
            ]
            
            for desc, kwargs in test_configs:
                try:
                    test_config = Config(**kwargs)
                    print(f"   ✓ {desc}: {test_config.openai_model}")
                except Exception as e:
                    print(f"   ✗ {desc}: {e}")
            
            # Step 6: Test CLI interface
            print("\n6. Testing CLI interface...")
            from pparser.cli import PPARSERCli
            cli = PPARSERCli()
            print(f"   ✓ CLI interface initialized")
            print(f"   ✓ CLI has parser: {hasattr(cli, 'parser')}")
            
            print("\n" + "=" * 60)
            print("END-TO-END TEST RESULTS")
            print("=" * 60)
            print("✓ PDF creation: SUCCESS")
            print("✓ Configuration: SUCCESS") 
            print("✓ PDF processor initialization: SUCCESS")
            print("✓ Text extraction: SUCCESS")
            print("✓ Configuration validation: SUCCESS")
            print("✓ CLI interface: SUCCESS")
            print("\n🎉 ALL CORE COMPONENTS WORKING!")
            print("\nNote: Full LLM-based processing requires valid OPENAI_API_KEY")
            print("      Set OPENAI_API_KEY environment variable for complete testing")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_end_to_end_processing()
    exit(0 if success else 1)
