"""
Batch processing workflow for multiple PDF files.

This module implements batch processing capabilities using the PDFWorkflow
for processing multiple PDF files concurrently.
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import Config
from ..utils.logger import get_logger
from .pdf_workflow import PDFWorkflow

logger = get_logger(__name__)


class BatchWorkflow:
    """
    Batch processing workflow for multiple PDF files.
    
    This class manages the concurrent processing of multiple PDF files
    using the PDFWorkflow for individual file processing.
    """

    def __init__(self, config: Optional[Config] = None, max_workers: int = 4):
        """
        Initialize the batch workflow.
        
        Args:
            config: Configuration object. If None, uses default config.
            max_workers: Maximum number of concurrent workers for processing.
        """
        self.config = config or Config()
        self.max_workers = max_workers
        self.logger = logger
        
        # Initialize PDF workflow template
        self.pdf_workflow = PDFWorkflow(self.config)

    async def process_directory(
        self, 
        input_dir: str, 
        output_dir: str,
        file_pattern: str = "*.pdf",
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files to process
            output_dir: Directory to save output files
            file_pattern: Glob pattern for PDF files (default: "*.pdf")
            recursive: Whether to search subdirectories recursively
            
        Returns:
            Dictionary containing batch processing results
        """
        self.logger.info(f"Starting batch processing for directory: {input_dir}")
        
        # Find PDF files
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if recursive:
            pdf_files = list(input_path.rglob(file_pattern))
        else:
            pdf_files = list(input_path.glob(file_pattern))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {input_dir} with pattern {file_pattern}")
            return {
                'status': 'completed',
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'results': [],
                'errors': []
            }
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process files
        return await self.process_files(
            [str(f) for f in pdf_files], 
            output_dir
        )

    async def process_files(
        self, 
        pdf_files: List[str], 
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Process a list of PDF files.
        
        Args:
            pdf_files: List of PDF file paths to process
            output_dir: Directory to save output files
            
        Returns:
            Dictionary containing batch processing results
        """
        self.logger.info(f"Processing {len(pdf_files)} PDF files")
        
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        errors = []
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process files with controlled concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_file(pdf_file: str) -> Dict[str, Any]:
            """Process a single PDF file with semaphore control."""
            async with semaphore:
                try:
                    # Create individual output directory for each file
                    file_stem = Path(pdf_file).stem
                    file_output_dir = output_path / file_stem
                    
                    # Use a fresh workflow instance for each file
                    workflow = PDFWorkflow(self.config)
                    result = await workflow.process_pdf(pdf_file, str(file_output_dir))
                    
                    result['pdf_file'] = pdf_file
                    result['processing_time'] = time.time() - start_time
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Error processing {pdf_file}: {str(e)}")
                    return {
                        'pdf_file': pdf_file,
                        'status': 'error',
                        'errors': [f"Processing error: {str(e)}"],
                        'processing_time': time.time() - start_time
                    }
        
        # Execute processing tasks
        tasks = [process_single_file(pdf_file) for pdf_file in pdf_files]
        
        # Process with progress tracking
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            try:
                result = await task
                results.append(result)
                
                if result['status'] in ['completed', 'validated']:
                    successful += 1
                    self.logger.info(f"Successfully processed {result['pdf_file']} ({i}/{len(pdf_files)})")
                else:
                    failed += 1
                    errors.extend(result.get('errors', []))
                    self.logger.error(f"Failed to process {result['pdf_file']} ({i}/{len(pdf_files)})")
                    
            except Exception as e:
                failed += 1
                error_msg = f"Task execution error: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        total_time = time.time() - start_time
        
        # Log summary
        self.logger.info(f"Batch processing completed in {total_time:.2f} seconds")
        self.logger.info(f"Results: {successful} successful, {failed} failed out of {len(pdf_files)} files")
        
        return {
            'status': 'completed',
            'total_files': len(pdf_files),
            'successful': successful,
            'failed': failed,
            'processing_time': total_time,
            'results': results,
            'errors': errors,
            'summary': {
                'success_rate': (successful / len(pdf_files)) * 100 if pdf_files else 0,
                'average_time_per_file': total_time / len(pdf_files) if pdf_files else 0,
                'total_errors': len(errors)
            }
        }

    async def process_with_retry(
        self, 
        pdf_files: List[str], 
        output_dir: str,
        max_retries: int = 2,
        retry_delay: float = 5.0
    ) -> Dict[str, Any]:
        """
        Process PDF files with retry mechanism for failed files.
        
        Args:
            pdf_files: List of PDF file paths to process
            output_dir: Directory to save output files
            max_retries: Maximum number of retry attempts for failed files
            retry_delay: Delay between retry attempts in seconds
            
        Returns:
            Dictionary containing batch processing results with retry information
        """
        self.logger.info(f"Processing {len(pdf_files)} PDF files with retry (max_retries={max_retries})")
        
        # Initial processing
        result = await self.process_files(pdf_files, output_dir)
        
        # Identify failed files for retry
        failed_files = [
            r['pdf_file'] for r in result['results'] 
            if r['status'] not in ['completed', 'validated']
        ]
        
        retry_count = 0
        retry_results = []
        
        while failed_files and retry_count < max_retries:
            retry_count += 1
            self.logger.info(f"Retry attempt {retry_count}/{max_retries} for {len(failed_files)} failed files")
            
            # Wait before retry
            await asyncio.sleep(retry_delay)
            
            # Retry failed files
            retry_result = await self.process_files(failed_files, output_dir)
            retry_results.append(retry_result)
            
            # Update failed files list
            failed_files = [
                r['pdf_file'] for r in retry_result['results'] 
                if r['status'] not in ['completed', 'validated']
            ]
            
            # Update overall results
            for retry_res in retry_result['results']:
                # Find and update the original result
                for i, orig_res in enumerate(result['results']):
                    if orig_res['pdf_file'] == retry_res['pdf_file']:
                        result['results'][i] = retry_res
                        break
        
        # Recalculate final statistics
        successful = sum(1 for r in result['results'] if r['status'] in ['completed', 'validated'])
        failed = len(result['results']) - successful
        
        result.update({
            'successful': successful,
            'failed': failed,
            'retry_attempts': retry_count,
            'retry_results': retry_results,
            'summary': {
                'success_rate': (successful / len(pdf_files)) * 100 if pdf_files else 0,
                'final_failed_files': [r['pdf_file'] for r in result['results'] if r['status'] not in ['completed', 'validated']],
                'retry_improved': len(pdf_files) - len(failed_files) if retry_count > 0 else 0
            }
        })
        
        self.logger.info(f"Batch processing with retry completed: {successful}/{len(pdf_files)} successful")
        
        return result

    def generate_batch_report(self, results: Dict[str, Any], output_path: str) -> str:
        """
        Generate a detailed batch processing report.
        
        Args:
            results: Results from batch processing
            output_path: Path to save the report
            
        Returns:
            Path to the generated report file
        """
        report_file = Path(output_path) / "batch_processing_report.md"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# Batch PDF Processing Report\n\n")
                
                # Summary section
                f.write("## Summary\n\n")
                f.write(f"- **Total Files**: {results['total_files']}\n")
                f.write(f"- **Successful**: {results['successful']}\n")
                f.write(f"- **Failed**: {results['failed']}\n")
                f.write(f"- **Success Rate**: {results['summary']['success_rate']:.1f}%\n")
                f.write(f"- **Total Processing Time**: {results['processing_time']:.2f} seconds\n")
                f.write(f"- **Average Time per File**: {results['summary']['average_time_per_file']:.2f} seconds\n\n")
                
                # Retry information if available
                if 'retry_attempts' in results:
                    f.write(f"- **Retry Attempts**: {results['retry_attempts']}\n")
                    f.write(f"- **Files Improved by Retry**: {results['summary'].get('retry_improved', 0)}\n\n")
                
                # Detailed results
                f.write("## Detailed Results\n\n")
                
                # Successful files
                successful_files = [r for r in results['results'] if r['status'] in ['completed', 'validated']]
                if successful_files:
                    f.write("### Successfully Processed Files\n\n")
                    for result in successful_files:
                        f.write(f"- **{Path(result['pdf_file']).name}**\n")
                        f.write(f"  - Status: {result['status']}\n")
                        f.write(f"  - Processing Time: {result.get('processing_time', 0):.2f}s\n")
                        if 'output_files' in result and result['output_files']:
                            f.write(f"  - Output Files: {len(result['output_files'])}\n")
                        f.write("\n")
                
                # Failed files
                failed_files = [r for r in results['results'] if r['status'] not in ['completed', 'validated']]
                if failed_files:
                    f.write("### Failed Files\n\n")
                    for result in failed_files:
                        f.write(f"- **{Path(result['pdf_file']).name}**\n")
                        f.write(f"  - Status: {result['status']}\n")
                        if result.get('errors'):
                            f.write(f"  - Errors:\n")
                            for error in result['errors']:
                                f.write(f"    - {error}\n")
                        f.write("\n")
                
                # Error summary
                if results.get('errors'):
                    f.write("## Error Summary\n\n")
                    for error in results['errors']:
                        f.write(f"- {error}\n")
                    f.write("\n")
                
                f.write("---\n")
                f.write(f"*Report generated by PPARSER Batch Processor*\n")
            
            self.logger.info(f"Batch report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Error generating batch report: {str(e)}")
            return ""
