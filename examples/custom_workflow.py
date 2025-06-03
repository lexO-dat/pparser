#!/usr/bin/env python3
"""
Custom Workflow Example

This example demonstrates how to create and use custom workflows.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pparser.workflows import PDFWorkflow
from pparser.config import Config
from pparser.utils.logger import setup_logging


async def custom_workflow_example():
    """Demonstrate custom workflow usage."""
    
    # Set up logging
    setup_logging(level="INFO")
    
    # Load configuration
    config = Config()
    
    # Create workflow
    workflow = PDFWorkflow()
    
    # Example PDF path
    pdf_path = "sample.pdf"
    output_dir = "workflow_output"
    
    print(f"Custom Workflow Example")
    print(f"Input: {pdf_path}")
    print(f"Output: {output_dir}")
    print()
    
    try:
        # Check if PDF exists
        if not Path(pdf_path).exists():
            print(f"PDF file not found: {pdf_path}")
            print("Please place a PDF file named 'sample.pdf' in the current directory")
            return False
        
        # Prepare workflow state
        initial_state = {
            "pdf_path": pdf_path,
            "output_dir": output_dir,
            "config": config,
            "extracted_content": {},
            "analyzed_content": {},
            "markdown_content": "",
            "quality_report": {},
            "assets": [],
            "processing_errors": [],
            "current_stage": "initialization"
        }
        
        print(f"Starting custom workflow...")
        
        # Execute workflow
        final_state = await workflow.workflow.ainvoke(initial_state)
        
        # Check results
        if final_state.get("success", False):
            print(f"Workflow completed successfully!")
            print(f"Markdown generated: {bool(final_state.get('markdown_content'))}")
            print(f"Assets created: {len(final_state.get('assets', []))}")
            print(f"Quality score: {final_state.get('quality_score', 'N/A')}")
            
            # Show processing stages
            if final_state.get("processing_stages"):
                print(f"\nProcessing stages completed:")
                for stage in final_state["processing_stages"]:
                    print(f"   {stage}")
            
            return True
        else:
            print(f"Workflow failed")
            errors = final_state.get("processing_errors", [])
            if errors:
                print(f"Errors:")
                for error in errors:
                    print(f"   • {error}")
            return False
            
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        return False


async def workflow_visualization():
    """Display workflow structure."""
    
    print(f"Workflow Visualization")
    print("-" * 30)
    
    try:
        # Create workflow
        workflow = PDFWorkflow()
        
        # Generate visualization
        from pparser.processors import PDFProcessor
        config = Config()
        processor = PDFProcessor(config)
        
        mermaid_diagram = processor.get_workflow_visualization()
        
        print("Mermaid Diagram (copy to https://mermaid.live for visualization):")
        print()
        print(mermaid_diagram)
        print()
        
        return True
        
    except Exception as e:
        print(f"Error generating visualization: {e}")
        return False


def main():
    """Main function."""
    print("PPARSER - Custom Workflow Example")
    print("=" * 50)
    
    try:
        # Show workflow visualization first
        print("1. Workflow Visualization")
        print("-" * 25)
        asyncio.run(workflow_visualization())
        
        print("\n2. Custom Workflow Execution")
        print("-" * 30)
        result = asyncio.run(custom_workflow_example())
        
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
