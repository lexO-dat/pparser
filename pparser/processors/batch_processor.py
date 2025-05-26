"""
Batch processor for handling multiple PDF files.

This module provides high-level batch processing capabilities using the
PDFProcessor and BatchWorkflow classes.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio
import time

from ..config import Config
from ..utils.logger import get_logger
from ..workflows.batch_workflow import BatchWorkflow
from .pdf_processor import PDFProcessor

logger = get_logger(__name__)


class BatchProcessor:
    """
    High-level batch processor for converting multiple PDF files to Markdown.
    
    This class provides an easy-to-use interface for batch processing operations
    with progress tracking, error handling, and comprehensive reporting.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        max_workers: int = 4,
        enable_quality_check: bool = True
    ):
        """
        Initialize the batch processor.
        
        Args:
            config: Configuration object. If None, uses default config.
            max_workers: Maximum number of concurrent workers for processing.
            enable_quality_check: Whether to enable quality validation by default.
        """
        self.config = config or Config()
        self.max_workers = max_workers
        self.enable_quality_check = enable_quality_check
        self.logger = logger
        
        # Initialize components
        self.batch_workflow = BatchWorkflow(self.config, max_workers)
        self.pdf_processor = PDFProcessor(self.config)

    async def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.pdf",
        recursive: bool = True,
        quality_check: Optional[bool] = None,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files to process
            output_dir: Directory to save output files
            file_pattern: Glob pattern for PDF files (default: "*.pdf")
            recursive: Whether to search subdirectories recursively
            quality_check: Override default quality check setting
            generate_report: Whether to generate a detailed batch report
            
        Returns:
            Dictionary containing comprehensive batch processing results
        """
        start_time = time.time()
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Validate inputs
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting batch processing: {input_dir} -> {output_dir}")
        self.logger.info(f"Settings: pattern={file_pattern}, recursive={recursive}, workers={self.max_workers}")
        
        # Use provided quality_check setting or default
        quality_check = quality_check if quality_check is not None else self.enable_quality_check
        
        try:
            # Find PDF files
            if recursive:
                pdf_files = list(input_path.rglob(file_pattern))
            else:
                pdf_files = list(input_path.glob(file_pattern))
            
            if not pdf_files:
                self.logger.warning(f"No PDF files found matching pattern: {file_pattern}")
                return self._create_empty_result(input_dir, output_dir, start_time)
            
            self.logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            # Process files with enhanced PDFProcessor
            results = await self._process_files_enhanced(
                [str(f) for f in pdf_files],
                str(output_path),
                quality_check
            )
            
            # Generate comprehensive report if requested
            if generate_report:
                report_path = await self._generate_comprehensive_report(results, str(output_path))
                results['report_file'] = report_path
            
            results.update({
                'input_directory': str(input_path),
                'output_directory': str(output_path),
                'total_processing_time': time.time() - start_time,
                'settings': {
                    'file_pattern': file_pattern,
                    'recursive': recursive,
                    'max_workers': self.max_workers,
                    'quality_check_enabled': quality_check
                }
            })
            
            self.logger.info(f"Batch processing completed: {results['successful']}/{results['total_files']} successful")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'input_directory': str(input_path),
                'output_directory': str(output_path),
                'total_processing_time': time.time() - start_time
            }

    async def process_file_list(
        self,
        pdf_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        quality_check: Optional[bool] = None,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Process a specific list of PDF files.
        
        Args:
            pdf_files: List of PDF file paths to process
            output_dir: Directory to save output files
            quality_check: Override default quality check setting
            generate_report: Whether to generate a detailed batch report
            
        Returns:
            Dictionary containing comprehensive batch processing results
        """
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Validate PDF files
        valid_files = []
        invalid_files = []
        
        for pdf_file in pdf_files:
            pdf_path = Path(pdf_file)
            if pdf_path.exists() and pdf_path.suffix.lower() == '.pdf':
                valid_files.append(str(pdf_path))
            else:
                invalid_files.append(str(pdf_file))
        
        if invalid_files:
            self.logger.warning(f"Found {len(invalid_files)} invalid PDF files: {invalid_files}")
        
        if not valid_files:
            self.logger.error("No valid PDF files to process")
            return self._create_empty_result("file_list", output_dir, start_time)
        
        self.logger.info(f"Processing {len(valid_files)} valid PDF files")
        
        # Use provided quality_check setting or default
        quality_check = quality_check if quality_check is not None else self.enable_quality_check
        
        try:
            # Process files with enhanced PDFProcessor
            results = await self._process_files_enhanced(
                valid_files,
                str(output_path),
                quality_check
            )
            
            # Add invalid files to results
            if invalid_files:
                results['invalid_files'] = invalid_files
                results['errors'].extend([f"Invalid file: {f}" for f in invalid_files])
            
            # Generate comprehensive report if requested
            if generate_report:
                report_path = await self._generate_comprehensive_report(results, str(output_path))
                results['report_file'] = report_path
            
            results.update({
                'input_source': 'file_list',
                'output_directory': str(output_path),
                'total_processing_time': time.time() - start_time,
                'settings': {
                    'max_workers': self.max_workers,
                    'quality_check_enabled': quality_check,
                    'total_input_files': len(pdf_files),
                    'valid_files': len(valid_files),
                    'invalid_files': len(invalid_files)
                }
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"File list processing failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'input_source': 'file_list',
                'output_directory': str(output_path),
                'total_processing_time': time.time() - start_time
            }

    async def _process_files_enhanced(
        self,
        pdf_files: List[str],
        output_dir: str,
        quality_check: bool
    ) -> Dict[str, Any]:
        """
        Process files using enhanced PDFProcessor with quality validation.
        
        Args:
            pdf_files: List of PDF file paths
            output_dir: Output directory
            quality_check: Whether to perform quality validation
            
        Returns:
            Enhanced processing results with quality metrics
        """
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        errors = []
        quality_scores = []
        
        # Create semaphore for controlled concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_file(pdf_file: str) -> Dict[str, Any]:
            """Process a single PDF file with quality validation."""
            async with semaphore:
                try:
                    # Create individual output directory
                    file_stem = Path(pdf_file).stem
                    file_output_dir = Path(output_dir) / file_stem
                    
                    # Process with quality validation
                    result = await self.pdf_processor.process(
                        pdf_file,
                        file_output_dir,
                        quality_check=quality_check,
                        return_metadata=True
                    )
                    
                    result['processing_time'] = time.time() - start_time
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Error processing {pdf_file}: {str(e)}")
                    return {
                        'pdf_file': pdf_file,
                        'status': 'error',
                        'errors': [f"Processing error: {str(e)}"],
                        'processing_time': time.time() - start_time,
                        'quality_score': 0
                    }
        
        # Execute processing tasks with progress tracking
        tasks = [process_single_file(pdf_file) for pdf_file in pdf_files]
        
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            try:
                result = await task
                results.append(result)
                
                if result['status'] in ['completed', 'high_quality', 'acceptable_quality']:
                    successful += 1
                    quality_score = result.get('quality_score', 0)
                    quality_scores.append(quality_score)
                    self.logger.info(f"Successfully processed {result['pdf_file']} (Quality: {quality_score:.1f}) ({i}/{len(pdf_files)})")
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
        
        # Calculate quality statistics
        quality_stats = self._calculate_quality_statistics(quality_scores)
        
        return {
            'status': 'completed',
            'total_files': len(pdf_files),
            'successful': successful,
            'failed': failed,
            'processing_time': total_time,
            'results': results,
            'errors': errors,
            'quality_statistics': quality_stats,
            'summary': {
                'success_rate': (successful / len(pdf_files)) * 100 if pdf_files else 0,
                'average_time_per_file': total_time / len(pdf_files) if pdf_files else 0,
                'total_errors': len(errors),
                'quality_enabled': quality_check
            }
        }

    async def _generate_comprehensive_report(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> str:
        """
        Generate a comprehensive batch processing report with quality metrics.
        
        Args:
            results: Batch processing results
            output_dir: Output directory for the report
            
        Returns:
            Path to the generated report file
        """
        report_file = Path(output_dir) / "comprehensive_batch_report.md"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# Comprehensive PDF Batch Processing Report\n\n")
                
                # Executive Summary
                f.write("## Executive Summary\n\n")
                f.write(f"- **Total Files Processed**: {results['total_files']}\n")
                f.write(f"- **Successful Conversions**: {results['successful']}\n")
                f.write(f"- **Failed Conversions**: {results['failed']}\n")
                f.write(f"- **Success Rate**: {results['summary']['success_rate']:.1f}%\n")
                f.write(f"- **Total Processing Time**: {results['processing_time']:.2f} seconds\n")
                f.write(f"- **Average Time per File**: {results['summary']['average_time_per_file']:.2f} seconds\n\n")
                
                # Quality Statistics
                quality_stats = results.get('quality_statistics', {})
                if quality_stats:
                    f.write("## Quality Assessment\n\n")
                    f.write(f"- **Average Quality Score**: {quality_stats.get('average_score', 0):.1f}/100\n")
                    f.write(f"- **Highest Quality Score**: {quality_stats.get('max_score', 0):.1f}/100\n")
                    f.write(f"- **Lowest Quality Score**: {quality_stats.get('min_score', 0):.1f}/100\n")
                    f.write(f"- **High Quality Files** (≥80): {quality_stats.get('high_quality_count', 0)}\n")
                    f.write(f"- **Acceptable Quality Files** (60-79): {quality_stats.get('acceptable_quality_count', 0)}\n")
                    f.write(f"- **Low Quality Files** (<60): {quality_stats.get('low_quality_count', 0)}\n\n")
                
                # Processing Settings
                settings = results.get('settings', {})
                if settings:
                    f.write("## Processing Settings\n\n")
                    for key, value in settings.items():
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                    f.write("\n")
                
                # Detailed Results
                f.write("## Detailed Results\n\n")
                
                # High quality results
                high_quality_files = [
                    r for r in results['results'] 
                    if r.get('quality_score', 0) >= 80 and r['status'] in ['completed', 'high_quality', 'acceptable_quality']
                ]
                
                if high_quality_files:
                    f.write("### High Quality Conversions (≥80/100)\n\n")
                    for result in sorted(high_quality_files, key=lambda x: x.get('quality_score', 0), reverse=True):
                        f.write(f"- **{Path(result['pdf_file']).name}**\n")
                        f.write(f"  - Quality Score: {result.get('quality_score', 0):.1f}/100\n")
                        f.write(f"  - Processing Time: {result.get('processing_time', 0):.2f}s\n")
                        f.write(f"  - Status: {result['status']}\n")
                        if result.get('output_files'):
                            f.write(f"  - Output Files: {len(result['output_files'])}\n")
                        f.write("\n")
                
                # Acceptable quality results
                acceptable_quality_files = [
                    r for r in results['results'] 
                    if 60 <= r.get('quality_score', 0) < 80 and r['status'] in ['completed', 'acceptable_quality']
                ]
                
                if acceptable_quality_files:
                    f.write("### Acceptable Quality Conversions (60-79/100)\n\n")
                    for result in sorted(acceptable_quality_files, key=lambda x: x.get('quality_score', 0), reverse=True):
                        f.write(f"- **{Path(result['pdf_file']).name}**\n")
                        f.write(f"  - Quality Score: {result.get('quality_score', 0):.1f}/100\n")
                        f.write(f"  - Processing Time: {result.get('processing_time', 0):.2f}s\n")
                        f.write(f"  - Status: {result['status']}\n")
                        f.write("\n")
                
                # Low quality and failed results
                problematic_files = [
                    r for r in results['results'] 
                    if r.get('quality_score', 0) < 60 or r['status'] not in ['completed', 'high_quality', 'acceptable_quality']
                ]
                
                if problematic_files:
                    f.write("### Problematic Conversions\n\n")
                    for result in problematic_files:
                        f.write(f"- **{Path(result['pdf_file']).name}**\n")
                        f.write(f"  - Quality Score: {result.get('quality_score', 0):.1f}/100\n")
                        f.write(f"  - Status: {result['status']}\n")
                        if result.get('errors'):
                            f.write(f"  - Errors:\n")
                            for error in result['errors'][:3]:  # Limit to first 3 errors
                                f.write(f"    - {error}\n")
                        f.write("\n")
                
                # Error Summary
                if results.get('errors'):
                    f.write("## Error Summary\n\n")
                    error_counts = {}
                    for error in results['errors']:
                        error_type = error.split(':')[0] if ':' in error else 'General'
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    
                    for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"- **{error_type}**: {count} occurrences\n")
                    f.write("\n")
                
                # Recommendations
                f.write("## Recommendations\n\n")
                success_rate = results['summary']['success_rate']
                avg_quality = quality_stats.get('average_score', 0)
                
                if success_rate < 80:
                    f.write("- **Low Success Rate**: Consider checking PDF file quality and format compatibility\n")
                if avg_quality < 70:
                    f.write("- **Low Average Quality**: Review content extraction settings and consider manual validation\n")
                if results['failed'] > 0:
                    f.write("- **Failed Files**: Investigate common error patterns and adjust processing parameters\n")
                if success_rate >= 90 and avg_quality >= 80:
                    f.write("- **Excellent Results**: Processing pipeline is working optimally\n")
                
                f.write("\n---\n")
                f.write(f"*Report generated by PPARSER Batch Processor on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
            
            self.logger.info(f"Comprehensive batch report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            return ""

    def _calculate_quality_statistics(self, quality_scores: List[float]) -> Dict[str, Any]:
        """Calculate quality statistics from a list of quality scores."""
        if not quality_scores:
            return {
                'average_score': 0,
                'max_score': 0,
                'min_score': 0,
                'high_quality_count': 0,
                'acceptable_quality_count': 0,
                'low_quality_count': 0,
                'total_scored_files': 0
            }
        
        return {
            'average_score': sum(quality_scores) / len(quality_scores),
            'max_score': max(quality_scores),
            'min_score': min(quality_scores),
            'high_quality_count': sum(1 for score in quality_scores if score >= 80),
            'acceptable_quality_count': sum(1 for score in quality_scores if 60 <= score < 80),
            'low_quality_count': sum(1 for score in quality_scores if score < 60),
            'total_scored_files': len(quality_scores)
        }

    def _create_empty_result(self, input_source: str, output_dir: str, start_time: float) -> Dict[str, Any]:
        """Create an empty result structure for when no files are found."""
        return {
            'status': 'completed',
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'processing_time': time.time() - start_time,
            'results': [],
            'errors': [],
            'quality_statistics': self._calculate_quality_statistics([]),
            'summary': {
                'success_rate': 0,
                'average_time_per_file': 0,
                'total_errors': 0,
                'quality_enabled': self.enable_quality_check
            },
            'input_source': input_source,
            'output_directory': str(output_dir)
        }

    async def process_with_retry(
        self,
        pdf_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        max_retries: int = 2,
        retry_delay: float = 5.0
    ) -> Dict[str, Any]:
        """
        Process PDF files with automatic retry for failed conversions.
        
        Args:
            pdf_files: List of PDF file paths to process
            output_dir: Directory to save output files
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
            
        Returns:
            Dictionary containing processing results with retry information
        """
        self.logger.info(f"Starting batch processing with retry (max_retries={max_retries})")
        
        # Initial processing
        result = await self.process_file_list(pdf_files, output_dir, generate_report=False)
        
        # Identify failed files for retry
        failed_files = [
            r['pdf_file'] for r in result.get('results', [])
            if r['status'] not in ['completed', 'high_quality', 'acceptable_quality']
        ]
        
        retry_count = 0
        retry_results = []
        
        while failed_files and retry_count < max_retries:
            retry_count += 1
            self.logger.info(f"Retry attempt {retry_count}/{max_retries} for {len(failed_files)} failed files")
            
            # Wait before retry
            await asyncio.sleep(retry_delay)
            
            # Retry failed files
            retry_result = await self.process_file_list(failed_files, output_dir, generate_report=False)
            retry_results.append(retry_result)
            
            # Update failed files list
            failed_files = [
                r['pdf_file'] for r in retry_result.get('results', [])
                if r['status'] not in ['completed', 'high_quality', 'acceptable_quality']
            ]
            
            # Update overall results
            for retry_res in retry_result.get('results', []):
                # Find and update the original result
                for i, orig_res in enumerate(result.get('results', [])):
                    if orig_res['pdf_file'] == retry_res['pdf_file']:
                        result['results'][i] = retry_res
                        break
        
        # Recalculate final statistics
        successful = sum(1 for r in result.get('results', []) if r['status'] in ['completed', 'high_quality', 'acceptable_quality'])
        failed = len(result.get('results', [])) - successful
        
        result.update({
            'successful': successful,
            'failed': failed,
            'retry_attempts': retry_count,
            'retry_results': retry_results,
            'final_failed_files': [r['pdf_file'] for r in result.get('results', []) if r['status'] not in ['completed', 'high_quality', 'acceptable_quality']]
        })
        
        # Generate final report with retry information
        report_path = await self._generate_comprehensive_report(result, str(output_dir))
        result['report_file'] = report_path
        
        self.logger.info(f"Batch processing with retry completed: {successful}/{len(pdf_files)} successful")
        
        return result
