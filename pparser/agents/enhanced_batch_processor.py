"""
Enhanced batch processor for handling multiple PDF files with advanced features.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from ..utils.logger import get_logger
from .error_handling import ErrorHandler, ErrorSeverity
from .enhanced_processor import EnhancedProcessor

logger = get_logger(__name__)


class EnhancedBatchProcessor:
    """Enhanced batch processor with retry, advanced error handling, and comprehensive reporting."""
    
    def __init__(self,
                 agent_factory: 'AgentFactory',
                 max_workers: int = 4,
                 enable_quality_check: bool = True,
                 enable_retry: bool = True,
                 retry_delay: float = 5.0):
        """Initialize enhanced batch processor."""
        self.agent_factory = agent_factory
        self.max_workers = max_workers
        self.enable_quality_check = enable_quality_check
        self.enable_retry = enable_retry
        self.retry_delay = retry_delay
        
        self.error_handler = ErrorHandler()
        
        logger.info(f"Enhanced batch processor initialized with {max_workers} workers")
    
    async def process_directory(self,
                               input_dir: Union[str, Path],
                               output_dir: Union[str, Path],
                               file_pattern: str = "*.pdf",
                               recursive: bool = True,
                               generate_report: bool = True) -> Dict[str, Any]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Input directory containing PDF files
            output_dir: Output directory for results
            file_pattern: File pattern to match (default: *.pdf)
            recursive: Whether to search subdirectories
            generate_report: Whether to generate detailed report
            
        Returns:
            Comprehensive batch processing results
        """
        start_time = time.time()
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting enhanced batch processing: {input_dir} -> {output_dir}")
        
        try:
            # Find PDF files
            if recursive:
                pdf_files = list(input_path.rglob(file_pattern))
            else:
                pdf_files = list(input_path.glob(file_pattern))
            
            if not pdf_files:
                logger.warning(f"No PDF files found matching pattern: {file_pattern}")
                return self._create_empty_result(input_dir, output_dir, start_time)
            
            logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            # Process files
            results = await self._process_files_enhanced(
                [str(f) for f in pdf_files],
                str(output_path)
            )
            
            # Generate report if requested
            if generate_report:
                report_path = await self._generate_comprehensive_report(results, str(output_path))
                results['report_file'] = report_path
            
            # Add processing metadata
            results.update({
                'input_directory': str(input_path),
                'output_directory': str(output_path),
                'total_processing_time': time.time() - start_time,
                'settings': {
                    'file_pattern': file_pattern,
                    'recursive': recursive,
                    'max_workers': self.max_workers,
                    'quality_check_enabled': self.enable_quality_check
                }
            })
            
            logger.info(f"Enhanced batch processing completed: {results['successful']}/{results['total_files']} successful")
            return results
            
        except Exception as e:
            error_msg = f"Enhanced batch processing failed: {str(e)}"
            self.error_handler.handle_error(e, ErrorSeverity.HIGH, error_msg)
            
            return {
                'status': 'error',
                'error': error_msg,
                'input_directory': str(input_path),
                'output_directory': str(output_path),
                'total_processing_time': time.time() - start_time
            }
    
    async def process_directory_with_retry(self,
                                          input_dir: Union[str, Path],
                                          output_dir: Union[str, Path],
                                          max_retries: int = 2,
                                          retry_delay: float = 5.0,
                                          **kwargs) -> Dict[str, Any]:
        """Process directory with retry capability for failed files."""
        
        # Initial processing
        result = await self.process_directory(input_dir, output_dir, **kwargs)
        
        if result['status'] == 'error':
            return result
        
        # Find failed files
        failed_files = [
            r['pdf_file'] for r in result.get('results', [])
            if r['status'] not in ['completed', 'high_quality', 'acceptable_quality']
        ]
        
        retry_count = 0
        retry_results = []
        
        while failed_files and retry_count < max_retries:
            retry_count += 1
            logger.info(f"Retry attempt {retry_count}/{max_retries} for {len(failed_files)} failed files")
            
            # Wait before retry
            await asyncio.sleep(retry_delay)
            
            # Retry failed files
            retry_result = await self._process_files_enhanced(failed_files, str(output_dir))
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
        successful = sum(1 for r in result.get('results', []) 
                        if r['status'] in ['completed', 'high_quality', 'acceptable_quality'])
        failed = len(result.get('results', [])) - successful
        
        result.update({
            'successful': successful,
            'failed': failed,
            'retry_attempts': retry_count,
            'retry_results': retry_results,
            'final_failed_files': [r['pdf_file'] for r in result.get('results', []) 
                                  if r['status'] not in ['completed', 'high_quality', 'acceptable_quality']]
        })
        
        logger.info(f"Enhanced batch processing with retry completed: {successful}/{len(result.get('results', []))} successful")
        return result
    
    async def _process_files_enhanced(self, pdf_files: List[str], output_dir: str) -> Dict[str, Any]:
        """Process files using enhanced processors with quality validation."""
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        errors = []
        quality_scores = []
        
        # Create semaphore for controlled concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_file(pdf_file: str) -> Dict[str, Any]:
            """Process a single PDF file with enhanced processor."""
            async with semaphore:
                try:
                    # Create individual output directory
                    file_stem = Path(pdf_file).stem
                    file_output_dir = Path(output_dir) / file_stem
                    
                    # Create enhanced processor for this file
                    processor = self.agent_factory.create_enhanced_processor(
                        quality_check=self.enable_quality_check,
                        enable_structure_building=True,
                        enable_comprehensive_validation=True,
                        enable_memory=False  # Disable memory for batch processing
                    )
                    
                    # Process with enhanced processor
                    result = await processor.process_pdf(pdf_file, str(file_output_dir))
                    
                    return result
                    
                except Exception as e:
                    error_msg = f"Error processing {pdf_file}: {str(e)}"
                    logger.error(error_msg)
                    return {
                        'pdf_file': pdf_file,
                        'status': 'error',
                        'errors': [error_msg],
                        'processing_time': 0,
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
                    logger.info(f"Successfully processed {result['pdf_file']} (Quality: {quality_score:.1f}) ({i}/{len(pdf_files)})")
                else:
                    failed += 1
                    errors.extend(result.get('errors', []))
                    logger.error(f"Failed to process {result['pdf_file']} ({i}/{len(pdf_files)})")
                    
            except Exception as e:
                failed += 1
                error_msg = f"Task execution error: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
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
                'quality_enabled': self.enable_quality_check
            }
        }
    
    def _calculate_quality_statistics(self, quality_scores: List[float]) -> Dict[str, Any]:
        """Calculate quality statistics from scores."""
        if not quality_scores:
            return {}
        
        return {
            'average_score': sum(quality_scores) / len(quality_scores),
            'max_score': max(quality_scores),
            'min_score': min(quality_scores),
            'high_quality_count': sum(1 for score in quality_scores if score >= 80),
            'acceptable_quality_count': sum(1 for score in quality_scores if 60 <= score < 80),
            'low_quality_count': sum(1 for score in quality_scores if score < 60),
            'total_scored_files': len(quality_scores)
        }
    
    def _create_empty_result(self, input_dir: str, output_dir: str, start_time: float) -> Dict[str, Any]:
        """Create empty result for when no files are found."""
        return {
            'status': 'completed',
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'processing_time': time.time() - start_time,
            'results': [],
            'errors': [],
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'summary': {
                'success_rate': 0,
                'average_time_per_file': 0,
                'total_errors': 0
            }
        }
    
    async def _generate_comprehensive_report(self, results: Dict[str, Any], output_dir: str) -> str:
        """Generate comprehensive batch processing report."""
        report_file = Path(output_dir) / "enhanced_batch_processing_report.md"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# Enhanced Batch PDF Processing Report\n\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary section
                f.write("## Summary\n\n")
                f.write(f"- **Total Files**: {results['total_files']}\n")
                f.write(f"- **Successful**: {results['successful']}\n")
                f.write(f"- **Failed**: {results['failed']}\n")
                f.write(f"- **Success Rate**: {results['summary']['success_rate']:.1f}%\n")
                f.write(f"- **Total Processing Time**: {results['processing_time']:.2f} seconds\n")
                f.write(f"- **Average Time per File**: {results['summary']['average_time_per_file']:.2f} seconds\n\n")
                
                # Quality statistics
                quality_stats = results.get('quality_statistics', {})
                if quality_stats:
                    f.write("## Quality Assessment\n\n")
                    f.write(f"- **Average Quality Score**: {quality_stats.get('average_score', 0):.1f}/100\n")
                    f.write(f"- **Highest Quality Score**: {quality_stats.get('max_score', 0):.1f}/100\n")
                    f.write(f"- **Lowest Quality Score**: {quality_stats.get('min_score', 0):.1f}/100\n")
                    f.write(f"- **High Quality Files** (≥80): {quality_stats.get('high_quality_count', 0)}\n")
                    f.write(f"- **Acceptable Quality Files** (60-79): {quality_stats.get('acceptable_quality_count', 0)}\n")
                    f.write(f"- **Low Quality Files** (<60): {quality_stats.get('low_quality_count', 0)}\n\n")
                
                # Processing settings
                settings = results.get('settings', {})
                if settings:
                    f.write("## Processing Settings\n\n")
                    for key, value in settings.items():
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                    f.write("\n")
                
                # Detailed results
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
                
                # Problematic results
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
                
                # Error summary
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
                f.write(f"*Report generated by PPARSER Enhanced Batch Processor*\n")
            
            logger.info(f"Enhanced batch report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error generating enhanced report: {str(e)}")
            return ""
