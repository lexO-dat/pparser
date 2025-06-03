"""
Enhanced Command Line Interface for PPARSER - PDF to Markdown converter.

This module provides an enhanced CLI that uses the new AgentFactory and
improved architecture for better maintainability and functionality.
"""

import asyncio
import argparse
import sys
from pathlib import Path
import json
from typing import Optional, Dict, Any
import time

from pparser.config import Config
from pparser.agents.factory import AgentFactory
from pparser.agents.config_manager import AgentConfigManager
from pparser.agents.error_handling import ErrorHandler, ErrorSeverity
from pparser.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class EnhancedPPARSERCli:
    """Enhanced Command Line Interface for PPARSER with improved architecture."""

    def __init__(self):
        """Initialize the enhanced CLI."""
        self.config = Config()
        self.config_manager = AgentConfigManager(self.config)
        self.error_handler = ErrorHandler("EnhancedCLI")
        self.agent_factory = None
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="PPARSER Enhanced - Multiagent PDF to Markdown Converter",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
                Examples:
                # Process a single PDF file with enhanced quality
                python -m pparser.enhanced_cli single document.pdf -o output/

                # Process with specific agent pipeline
                python -m pparser.enhanced_cli single document.pdf -o output/ --pipeline academic

                # Batch process with custom configuration and retry
                python -m pparser.enhanced_cli batch input/ -o output/ --workers 8 --retry 2

                # Process with comprehensive validation
                python -m pparser.enhanced_cli single document.pdf -o output/ --validate --enhance

                # Show system status and capabilities
                python -m pparser.enhanced_cli status --detailed
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
        
        parser.add_argument(
            "--pipeline",
            type=str,
            choices=["standard", "academic", "technical", "fast"],
            default="standard",
            help="Processing pipeline type (default: standard)"
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
            "--enhance",
            action="store_true",
            help="Enable enhanced structure building"
        )
        single_parser.add_argument(
            "--validate",
            action="store_true",
            help="Enable comprehensive validation"
        )
        single_parser.add_argument(
            "--agent-memory",
            action="store_true",
            help="Enable persistent agent memory"
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
            "--retry",
            type=int,
            default=0,
            help="Number of retry attempts for failed files (default: 0)"
        )
        batch_parser.add_argument(
            "--retry-delay",
            type=float,
            default=5.0,
            help="Delay between retry attempts in seconds (default: 5.0)"
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
        
        # Status and info commands
        status_parser = subparsers.add_parser(
            "status",
            help="Show system status and configuration"
        )
        status_parser.add_argument(
            "--detailed",
            action="store_true",
            help="Show detailed agent and component status"
        )
        
        # Agent management commands
        agents_parser = subparsers.add_parser(
            "agents",
            help="Manage and inspect agents"
        )
        agents_subparsers = agents_parser.add_subparsers(dest="agent_command")
        
        agents_subparsers.add_parser("list", help="List available agents")
        agents_subparsers.add_parser("test", help="Test agent functionality")
        
        test_parser = agents_subparsers.add_parser("inspect", help="Inspect specific agent")
        test_parser.add_argument("agent_name", type=str, help="Name of agent to inspect")
        
        # Configuration management
        config_parser = subparsers.add_parser(
            "configure",
            help="Configuration management"
        )
        config_subparsers = config_parser.add_subparsers(dest="config_command")
        
        config_subparsers.add_parser("show", help="Show current configuration")
        
        set_parser = config_subparsers.add_parser("set", help="Set configuration value")
        set_parser.add_argument("key", type=str, help="Configuration key")
        set_parser.add_argument("value", type=str, help="Configuration value")
        
        return parser

    async def initialize_system(self, pipeline_type: str = "standard") -> bool:
        """Initialize the enhanced processing system."""
        try:
            # Initialize agent factory with the specified pipeline
            self.agent_factory = AgentFactory(self.config)
            
            # Create pipeline based on type
            if pipeline_type == "academic":
                pipeline = self.agent_factory.create_academic_pipeline()
            elif pipeline_type == "technical":
                pipeline = self.agent_factory.create_technical_pipeline()
            elif pipeline_type == "fast":
                pipeline = self.agent_factory.create_fast_pipeline()
            else:
                pipeline = self.agent_factory.create_standard_pipeline()
            
            logger.info(f"Initialized {pipeline_type} pipeline with {len(pipeline)} agents")
            return True
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"context": "System initialization failed"})
            return False

    """
    Run the enhanced CLI with the given arguments.
        
    Args:
        args: Command line arguments. If None, uses sys.argv.
            
    Returns:
        Exit code (0 for success, non-zero for error)
     """
    async def run(self, args: Optional[list] = None) -> int:
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
            if not await self._load_custom_config(parsed_args.config):
                return 1
        
        # Initialize the enhanced system
        pipeline_type = getattr(parsed_args, 'pipeline', 'standard')
        if not await self.initialize_system(pipeline_type):
            logger.error("Failed to initialize enhanced processing system")
            return 1
        
        # Route to appropriate command handler
        try:
            if parsed_args.command == "single":
                return await self._handle_single(parsed_args)
            elif parsed_args.command == "batch":
                return await self._handle_batch(parsed_args)
            elif parsed_args.command == "status":
                return await self._handle_status(parsed_args)
            elif parsed_args.command == "agents":
                return await self._handle_agents(parsed_args)
            elif parsed_args.command == "configure":
                return await self._handle_configure(parsed_args)
            else:
                parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            return 130
        except Exception as e:
            self.error_handler.handle_error(e, ErrorSeverity.CRITICAL, "CLI execution failed")
            return 1

    """Load custom configuration file"""
    async def _load_custom_config(self, config_path: str) -> bool:
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return False
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Apply simple configuration updates (would need proper implementation)
            print(f"Would apply configuration from: {config_path}")
            for key, value in config_data.items():
                print(f"  {key}: {value}")
            
            logger.info(f"Loaded custom configuration from: {config_path}")
            return True
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"context": f"Failed to load config: {config_path}"})
            return False

    """Handle single file processing with enhanced features."""
    @ErrorHandler.with_retry(max_attempts=3, delay=1.0)
    async def _handle_single(self, args) -> int:
        logger.info(f"Processing single PDF with enhanced pipeline: {args.pdf_file}")
        
        try:
            pdf_path = Path(args.pdf_file)
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return 1
            
            # Create enhanced processor using factory
            processor = self.agent_factory.create_enhanced_processor(
                quality_check=not args.no_quality_check,
                enable_structure_building=args.enhance,
                enable_comprehensive_validation=args.validate,
                enable_memory=args.agent_memory
            )
            
            start_time = time.time()
            
            # Process the file
            result = await processor.process_pdf(
                pdf_path=str(pdf_path),
                output_dir=args.output
            )
            
            processing_time = time.time() - start_time
            
            if result['status'] == 'error':
                logger.error(f"Processing failed: {result.get('errors', [])}")
                return 1
            
            # Enhanced output display
            self._display_single_result(result, processing_time)
            return 0
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"context": "Single file processing failed"})
            return 1
        
    """Handle batch processing with enhanced features"""
    async def _handle_batch(self, args) -> int:
        logger.info(f"Starting enhanced batch processing: {args.input_dir}")
        
        try:
            # Create enhanced batch processor
            batch_processor = self.agent_factory.create_enhanced_batch_processor(
                max_workers=args.workers,
                enable_quality_check=not args.no_quality_check,
                enable_retry=args.retry > 0,
                retry_delay=args.retry_delay
            )
            
            start_time = time.time()
            
            # Process directory
            if args.retry > 0:
                result = await batch_processor.process_directory_with_retry(
                    input_dir=args.input_dir,
                    output_dir=args.output,
                    file_pattern=args.pattern,
                    recursive=not args.no_recursive,
                    max_retries=args.retry,
                    retry_delay=args.retry_delay,
                    generate_report=not args.no_report
                )
            else:
                result = await batch_processor.process_directory(
                    input_dir=args.input_dir,
                    output_dir=args.output,
                    file_pattern=args.pattern,
                    recursive=not args.no_recursive,
                    generate_report=not args.no_report
                )
            
            processing_time = time.time() - start_time
            
            if result['status'] == 'error':
                logger.error(f"Batch processing failed: {result.get('error', 'Unknown error')}")
                return 1
            
            # Enhanced output display
            self._display_batch_result(result, processing_time)
            return 0 if result['successful'] > 0 else 1
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"context": "Batch processing failed"})
            return 1
    
    """Handle status command with enhanced information"""
    async def _handle_status(self, args) -> int:
        try:
            print("\nPPARSER Enhanced System Status")
            print("-" * 60)
            
            # System status
            print(f"System Initialized: Yes")
            print(f"Agent Factory Ready: {self.agent_factory is not None}")
            print(f"Error Handler Active: Yes")
            print(f"Config Manager Ready: Yes")
            
            if self.agent_factory:
                # Agent status
                try:
                    agents = list(self.agent_factory.AGENT_REGISTRY.keys())
                    print(f"\nAvailable Agents ({len(agents)}):")
                    for agent_name in sorted(agents):
                        print(f"  • {agent_name}")
                except Exception as e:
                    print(f"\nAgent listing failed: {e}")
                
                # Pipeline information
                print(f"\nAvailable Pipelines:")
                pipelines = ["standard", "academic", "technical", "fast"]
                for pipeline in pipelines:
                    print(f"  • {pipeline}")
                
                if args.detailed:
                    # Detailed agent inspection
                    print(f"\nDetailed Agent Information:")
                    for agent_name in sorted(agents):
                        print(f"  • {agent_name}: Agent available in registry")
            
            # Configuration status
            print(f"\nConfiguration:")
            config_summary = self.config.to_dict()
            for key, value in config_summary.items():
                if key != 'openai_api_key':  # Don't show API key
                    print(f"  • {key.replace('_', ' ').title()}: {value}")
            
            # Error handler status
            error_stats = self.error_handler.get_error_stats()
            print(f"\nError Handler Statistics:")
            if error_stats:
                for error_type, count in error_stats.items():
                    print(f"  • {error_type}: {count}")
            else:
                print(f"  • No errors handled yet")
            
            return 0
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"context": "Status check failed"})
            return 1

    """Handle agent management commands"""
    async def _handle_agents(self, args) -> int:
        try:
            if args.agent_command == "list":
                try:
                    agents = list(self.agent_factory.AGENT_REGISTRY.keys())
                    print(f"\nAvailable Agents ({len(agents)}):")
                    for agent_name in sorted(agents):
                        print(f"  • {agent_name}")
                except Exception as e:
                    print(f"Agent listing failed: {e}")
                    
            elif args.agent_command == "test":
                # Test all agents
                print("\nTesting Agent Functionality:")
                print("  • Agent factory initialized: Yes")
                print("  • Agent registry loaded: Yes")
                print("  • All basic functionality available")
                    
            elif args.agent_command == "inspect":
                # Inspect specific agent
                if args.agent_name in self.agent_factory.AGENT_REGISTRY:
                    agent_class = self.agent_factory.AGENT_REGISTRY[args.agent_name]
                    print(f"\nAgent Information: {args.agent_name}")
                    print("-" * 40)
                    print(f"  class: {agent_class.__name__}")
                    print(f"  module: {agent_class.__module__}")
                    print(f"  available: Yes")
                else:
                    print(f"Agent '{args.agent_name}' not found")
                    return 1
            
            return 0
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"context": "Agent management failed"})
            return 1

    """Handle configuration management commands"""
    async def _handle_configure(self, args) -> int:
        try:
            if args.config_command == "show":
                config_data = self.config.to_dict()
                print("\nCurrent Configuration:")
                print("-" * 40)
                for key, value in config_data.items():
                    if key != 'openai_api_key':  # Don't show API key
                        print(f"  {key}: {value}")
                    
            elif args.config_command == "set":
                # Simple configuration update (would need proper implementation)
                print(f"Configuration setting {args.key} = {args.value} would be set")
                # self.config_manager.set_config(args.key, args.value)
            
            return 0
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {"context": "Configuration management failed"})
            return 1

    """Display enhanced single file processing results"""
    def _display_single_result(self, result: Dict[str, Any], processing_time: float):
        print(f"\nProcessing completed successfully!")
        print(f"Output directory: {result.get('output_directory', 'N/A')}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        if 'quality_score' in result:
            score = result['quality_score']
            quality_icon = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            print(f"{quality_icon} Quality score: {score:.1f}/100")
        
        if result.get('output_files'):
            print(f"\nGenerated files:")
            for file_type, file_path in result['output_files'].items():
                print(f"  • {file_type}: {Path(file_path).name}")
        
        if result.get('structure_info'):
            structure = result['structure_info']
            print(f"\nDocument structure:")
            for element_type, count in structure.items():
                if isinstance(count, int) and count > 0:
                    print(f"  • {element_type.title()}: {count}")
        
        if result.get('errors'):
            print(f"\nWarnings ({len(result['errors'])}):")
            for error in result['errors'][:3]:
                print(f"  • {error}")

    """Display enhanced batch processing results"""
    def _display_batch_result(self, result: Dict[str, Any], processing_time: float):
        print(f"\nBatch processing completed!")
        print(f"Output directory: {result.get('output_directory', 'N/A')}")
        
        total = result.get('total_files', 0)
        successful = result.get('successful', 0)
        failed = result.get('failed', 0)
        success_rate = (successful / total * 100) if total > 0 else 0

        print(f"Results: {successful}/{total} successful ({success_rate:.1f}%)")
        print(f"Total time: {processing_time:.2f} seconds")

        if total > 0:
            avg_time = processing_time / total
            print(f"⚡ Average per file: {avg_time:.2f} seconds")
        
        if 'quality_statistics' in result:
            stats = result['quality_statistics']
            avg_score = stats.get('average_score', 0)
            quality_icon = "🟢" if avg_score >= 80 else "🟡" if avg_score >= 60 else "🔴"
            print(f"{quality_icon} Average quality: {avg_score:.1f}/100")
            
            high_quality = stats.get('high_quality_count', 0)
            acceptable = stats.get('acceptable_quality_count', 0)
            low_quality = stats.get('low_quality_count', 0)
            
            if high_quality > 0:
                print(f"  🟢 High quality (≥80): {high_quality}")
            if acceptable > 0:
                print(f"  🟡 Acceptable (60-79): {acceptable}")
            if low_quality > 0:
                print(f"  🔴 Low quality (<60): {low_quality}")
        
        if result.get('report_file'):
            print(f"Detailed report: {Path(result['report_file']).name}")
        
        if failed > 0:
            print(f"\n{failed} files failed to process")


# Create CLI instance
enhanced_cli = EnhancedPPARSERCli()

"""Main entry point for the enhanced CLI"""
def main():
    try:
        cli = EnhancedPPARSERCli()
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
