"""
Command Line Interface for PPARSER - PDF to Markdown converter.

This module provides a comprehensive CLI for the multiagent PDF processing system.
"""

import asyncio
import argparse
import sys
from pathlib import Path
import json
from typing import Optional

from pparser.config import Config
from pparser.processors import PDFProcessor, BatchProcessor
from pparser.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class PPARSERCli:
    """Command Line Interface for PPARSER."""

    def __init__(self):
        """Initialize the CLI."""
        self.config = Config()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="PPARSER - Multiagent PDF to Markdown Converter",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process a single PDF file
  python -m pparser single document.pdf -o output/

  # Process all PDFs in a directory
  python -m pparser batch input_dir/ -o output_dir/

  # Process with quality validation disabled
  python -m pparser single document.pdf -o output/ --no-quality-check

  # Batch process with custom settings
  python -m pparser batch input/ -o output/ --workers 8 --pattern "*.pdf" --recursive

  # Process with custom configuration
  python -m pparser single document.pdf -o output/ --config custom_config.json
            """
        )
        
        # Common arguments
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        
        parser.add_argument(
            "--config",
            type=str,
            help="Path to custom configuration file (JSON)"
        )
        
        parser.add_argument(
            "--log-file",
            type=str,
            help="Log file path (default: console only)"
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Single file processing
        single_parser = subparsers.add_parser(
            "single",
            help="Process a single PDF file"
        )
        single_parser.add_argument(
            "pdf_file",
            type=str,
            help="Path to the PDF file to process"
        )
        single_parser.add_argument(
            "-o", "--output",
            type=str,
            required=True,
            help="Output directory for converted files"
        )
        single_parser.add_argument(
            "--no-quality-check",
            action="store_true",
            help="Disable quality validation"
        )
        single_parser.add_argument(
            "--no-metadata",
            action="store_true",
            help="Don't include detailed metadata in output"
        )
        
        # Batch processing
        batch_parser = subparsers.add_parser(
            "batch",
            help="Process multiple PDF files"
        )
        batch_parser.add_argument(
            "input_dir",
            type=str,
            help="Directory containing PDF files to process"
        )
        batch_parser.add_argument(
            "-o", "--output",
            type=str,
            required=True,
            help="Output directory for converted files"
        )
        batch_parser.add_argument(
            "--pattern",
            type=str,
            default="*.pdf",
            help="File pattern to match (default: *.pdf)"
        )
        batch_parser.add_argument(
            "--no-recursive",
            action="store_true",
            help="Don't search subdirectories"
        )
        batch_parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="Number of concurrent workers (default: 4)"
        )
        batch_parser.add_argument(
            "--no-quality-check",
            action="store_true",
            help="Disable quality validation"
        )
        batch_parser.add_argument(
            "--no-report",
            action="store_true",
            help="Don't generate batch processing report"
        )
        batch_parser.add_argument(
            "--retry",
            type=int,
            default=0,
            help="Number of retry attempts for failed files (default: 0)"
        )
        
        # File list processing
        filelist_parser = subparsers.add_parser(
            "filelist",
            help="Process a list of specific PDF files"
        )
        filelist_parser.add_argument(
            "file_list",
            type=str,
            help="Path to text file containing PDF file paths (one per line)"
        )
        filelist_parser.add_argument(
            "-o", "--output",
            type=str,
            required=True,
            help="Output directory for converted files"
        )
        filelist_parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="Number of concurrent workers (default: 4)"
        )
        filelist_parser.add_argument(
            "--no-quality-check",
            action="store_true",
            help="Disable quality validation"
        )
        filelist_parser.add_argument(
            "--no-report",
            action="store_true",
            help="Don't generate batch processing report"
        )
        
        # Status and info commands
        status_parser = subparsers.add_parser(
            "status",
            help="Show system status and configuration"
        )
        
        workflow_parser = subparsers.add_parser(
            "workflow",
            help="Display workflow visualization"
        )
        
        return parser

    async def run(self, args: Optional[list] = None) -> int:
        """
        Run the CLI with the given arguments.
        
        Args:
            args: Command line arguments. If None, uses sys.argv.
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Setup logging
        log_level = "DEBUG" if parsed_args.verbose else "INFO"
        setup_logging(
            level=log_level,
            log_file=parsed_args.log_file
        )
        
        # Load custom config if provided
        if parsed_args.config:
            try:
                config_path = Path(parsed_args.config)
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Apply config overrides
                    for key, value in config_data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                    
                    logger.info(f"Loaded custom configuration from: {config_path}")
                else:
                    logger.error(f"Configuration file not found: {config_path}")
                    return 1
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                return 1
        
        # Route to appropriate command handler
        try:
            if parsed_args.command == "single":
                return await self._handle_single(parsed_args)
            elif parsed_args.command == "batch":
                return await self._handle_batch(parsed_args)
            elif parsed_args.command == "filelist":
                return await self._handle_filelist(parsed_args)
            elif parsed_args.command == "status":
                return await self._handle_status(parsed_args)
            elif parsed_args.command == "workflow":
                return await self._handle_workflow(parsed_args)
            else:
                parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return 1

    async def _handle_single(self, args) -> int:
        """Handle single file processing."""
        logger.info(f"Processing single PDF: {args.pdf_file}")
        
        try:
            pdf_path = Path(args.pdf_file)
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return 1
            
            processor = PDFProcessor(self.config)
            
            result = await processor.process(
                pdf_path=args.pdf_file,
                output_dir=args.output,
                quality_check=not args.no_quality_check,
                return_metadata=not args.no_metadata
            )
            
            if result['status'] == 'error':
                logger.error(f"Processing failed: {result.get('errors', [])}")
                return 1
            
            # Print results
            print(f"\n✅ Processing completed successfully!")
            print(f"📁 Output directory: {result['output_directory']}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
            print(f"📊 Quality score: {result['quality_score']:.1f}/100")
            
            if result.get('output_files'):
                print("\n📄 Generated files:")
                for file_type, file_path in result['output_files'].items():
                    print(f"  • {file_type}: {Path(file_path).name}")
            
            if result.get('errors'):
                print(f"\n⚠️  Warnings: {len(result['errors'])}")
                for error in result['errors'][:3]:  # Show first 3 errors
                    print(f"  • {error}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Single file processing failed: {str(e)}")
            return 1

    async def _handle_batch(self, args) -> int:
        """Handle batch processing."""
        logger.info(f"Starting batch processing: {args.input_dir}")
        
        try:
            processor = BatchProcessor(
                config=self.config,
                max_workers=args.workers,
                enable_quality_check=not args.no_quality_check
            )
            
            if args.retry > 0:
                # Find PDF files first
                input_path = Path(args.input_dir)
                if args.no_recursive:
                    pdf_files = list(input_path.glob(args.pattern))
                else:
                    pdf_files = list(input_path.rglob(args.pattern))
                
                result = await processor.process_with_retry(
                    pdf_files=pdf_files,
                    output_dir=args.output,
                    max_retries=args.retry
                )
            else:
                result = await processor.process_directory(
                    input_dir=args.input_dir,
                    output_dir=args.output,
                    file_pattern=args.pattern,
                    recursive=not args.no_recursive,
                    generate_report=not args.no_report
                )
            
            if result['status'] == 'error':
                logger.error(f"Batch processing failed: {result.get('error', 'Unknown error')}")
                return 1
            
            # Print results summary
            print(f"\n✅ Batch processing completed!")
            print(f"📁 Output directory: {result['output_directory']}")
            print(f"📊 Results: {result['successful']}/{result['total_files']} successful ({result['summary']['success_rate']:.1f}%)")
            print(f"⏱️  Total time: {result['processing_time']:.2f} seconds")
            
            if 'quality_statistics' in result:
                stats = result['quality_statistics']
                print(f"🎯 Quality: {stats['average_score']:.1f}/100 average")
                print(f"   • High quality (≥80): {stats['high_quality_count']}")
                print(f"   • Acceptable (60-79): {stats['acceptable_quality_count']}")
                print(f"   • Low quality (<60): {stats['low_quality_count']}")
            
            if result.get('report_file'):
                print(f"📋 Detailed report: {Path(result['report_file']).name}")
            
            if result['failed'] > 0:
                print(f"\n⚠️  {result['failed']} files failed to process")
                if result.get('errors'):
                    print("Common errors:")
                    for error in result['errors'][:5]:  # Show first 5 errors
                        print(f"  • {error}")
            
            return 0 if result['successful'] > 0 else 1
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return 1

    async def _handle_filelist(self, args) -> int:
        """Handle file list processing."""
        logger.info(f"Processing file list: {args.file_list}")
        
        try:
            # Read file list
            file_list_path = Path(args.file_list)
            if not file_list_path.exists():
                logger.error(f"File list not found: {file_list_path}")
                return 1
            
            with open(file_list_path, 'r') as f:
                pdf_files = [line.strip() for line in f if line.strip()]
            
            if not pdf_files:
                logger.error("No PDF files found in file list")
                return 1
            
            processor = BatchProcessor(
                config=self.config,
                max_workers=args.workers,
                enable_quality_check=not args.no_quality_check
            )
            
            result = await processor.process_file_list(
                pdf_files=pdf_files,
                output_dir=args.output,
                generate_report=not args.no_report
            )
            
            if result['status'] == 'error':
                logger.error(f"File list processing failed: {result.get('error', 'Unknown error')}")
                return 1
            
            # Print results (similar to batch processing)
            print(f"\n✅ File list processing completed!")
            print(f"📁 Output directory: {result['output_directory']}")
            print(f"📊 Results: {result['successful']}/{result['total_files']} successful ({result['summary']['success_rate']:.1f}%)")
            
            return 0 if result['successful'] > 0 else 1
            
        except Exception as e:
            logger.error(f"File list processing failed: {str(e)}")
            return 1

    async def _handle_status(self, args) -> int:
        """Handle status command."""
        try:
            processor = PDFProcessor(self.config)
            status = processor.get_processing_status()
            
            print("\n🔧 PPARSER System Status")
            print("=" * 50)
            
            print(f"Processor Ready: {'✅' if status['processor_ready'] else '❌'}")
            print(f"Configuration Loaded: {'✅' if status['config_loaded'] else '❌'}")
            print(f"Workflow Initialized: {'✅' if status['workflow_initialized'] else '❌'}")
            
            print("\nAgents Status:")
            agents = status['agents_ready']
            for agent_name, ready in agents.items():
                status_icon = '✅' if ready else '❌'
                print(f"  • {agent_name.replace('_', ' ').title()}: {status_icon}")
            
            print("\nSupported Features:")
            features = status['supported_features']
            for feature_name, supported in features.items():
                status_icon = '✅' if supported else '❌'
                print(f"  • {feature_name.replace('_', ' ').title()}: {status_icon}")
            
            print("\nConfiguration:")
            print(f"  • LLM Model: {self.config.openai_model}")
            print(f"  • Max Tokens: {self.config.max_tokens}")
            print(f"  • Temperature: {self.config.temperature}")
            # print(f"  • Image DPI: {self.config.image_dpi}")
            # print(f"  • Max Image Size: {self.config.max_image_size}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Status check failed: {str(e)}")
            return 1

    async def _handle_workflow(self, args) -> int:
        """Handle workflow visualization command."""
        try:
            processor = PDFProcessor(self.config)
            mermaid_graph = processor.get_workflow_visualization()
            
            print("\n🔄 PPARSER Workflow Visualization")
            print("=" * 50)
            print("Mermaid Diagram (copy to https://mermaid.live for visualization):")
            print()
            print(mermaid_graph)
            print()
            
            return 0
            
        except Exception as e:
            logger.error(f"Workflow visualization failed: {str(e)}")
            return 1


# Create a global CLI instance
cli = PPARSERCli()


def process_single(pdf_path: str, output_dir: str, **kwargs) -> int:
    """Process a single PDF file (convenience function for testing)"""
    import asyncio
    
    # Simulate command line arguments
    class Args:
        def __init__(self, **kwargs):
            self.pdf_path = pdf_path
            self.output_dir = output_dir
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(**kwargs)
    return asyncio.run(cli._handle_single(args))


def process_batch(input_dir: str, output_dir: str, **kwargs) -> int:
    """Process multiple PDF files (convenience function for testing)"""
    import asyncio
    
    # Simulate command line arguments
    class Args:
        def __init__(self, **kwargs):
            self.input_dir = input_dir
            self.output_dir = output_dir
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(**kwargs)
    return asyncio.run(cli._handle_batch(args))


def process_filelist(filelist_path: str, output_dir: str, **kwargs) -> int:
    """Process PDFs from file list (convenience function for testing)"""
    import asyncio
    
    # Simulate command line arguments
    class Args:
        def __init__(self, **kwargs):
            self.filelist = filelist_path
            self.output_dir = output_dir
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(**kwargs)
    return asyncio.run(cli._handle_filelist(args))


def main():
    """Main entry point for the CLI."""
    cli = PPARSERCli()
    
    try:
        # Run the CLI in async context
        exit_code = asyncio.run(cli.run())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
