"""
Table analysis and formatting agent
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from .base import BaseAgent
from ..extractors import TableExtractor


class TableAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing and formatting tables"""
    
    def __init__(self, config, output_dir: Optional[Path] = None):
        super().__init__(
            config=config,
            name="TableAnalysisAgent",
            role="Analyze and format tables for Markdown conversion",
            temperature=0.1
        )
        self.extractor = TableExtractor(config=config, output_dir=output_dir)
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process PDF page and analyze tables"""
        
        pdf_path = Path(input_data.get('pdf_path'))
        page_num = input_data.get('page_num', 0)
        
        # Extract tables
        extraction_result = self.extractor.extract(pdf_path, page_num)
        
        if not extraction_result.get('tables'):
            return {
                'success': True,
                'result': extraction_result,
                'agent': self.name,
                'message': 'No tables found on this page'
            }
        
        # Enhance table analysis with LLM
        enhanced_result = self._enhance_table_analysis(extraction_result)
        
        return {
            'success': True,
            'result': enhanced_result,
            'agent': self.name
        }
    
    def _enhance_table_analysis(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance table analysis using LLM"""
        
        enhanced_tables = []
        
        for table_info in extraction_result.get('tables', []):
            enhanced_table = table_info.copy()
            
            # Analyze table content and structure
            analysis = self._analyze_table_content(table_info)
            enhanced_table.update(analysis)
            
            # Improve table formatting
            improved_markdown = self._improve_table_formatting(table_info)
            enhanced_table['improved_markdown'] = improved_markdown
            
            # Generate table summary
            summary = self._generate_table_summary(enhanced_table)
            enhanced_table['summary'] = summary
            
            enhanced_tables.append(enhanced_table)
        
        result = extraction_result.copy()
        result['tables'] = enhanced_tables
        
        return result
    
    def _analyze_table_content(self, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze table content and structure"""
        
        headers = table_info.get('headers', [])
        markdown = table_info.get('markdown', '')
        filepath = table_info.get('filepath', '')
        
        system_prompt = """You are an expert in data analysis and table structure. Analyze this table and provide insights about its content, structure, and purpose.

                        Analyze:
                        1. Data types in each column (numeric, text, categorical, date, etc.)
                        2. Table purpose (comparison, data listing, summary, reference, etc.)
                        3. Key insights or patterns in the data
                        4. Suggested improvements for clarity
                        5. Whether headers are descriptive enough

                        Return your analysis in JSON format:
                        {
                            "column_types": ["type1", "type2", ...],
                            "table_purpose": "description",
                            "key_insights": ["insight1", "insight2"],
                            "header_quality": "good|fair|poor",
                            "suggested_improvements": ["improvement1", "improvement2"],
                            "complexity": "simple|moderate|complex"
                        }"""
        
        user_content = f"""Analyze this table:

                        HEADERS: {', '.join(headers)}

                        TABLE CONTENT: {markdown[:1000]}{"..." if len(markdown) > 1000 else ""}

                        Provide your analysis."""
        
        messages = self._create_messages(system_prompt, user_content)
        llm_response = self._invoke_llm(messages)
        
        # Parse analysis response
        analysis = self._parse_analysis_response(llm_response)
        
        return analysis
    
    def _improve_table_formatting(self, table_info: Dict[str, Any]) -> str:
        """Improve table formatting using LLM"""
        
        original_markdown = table_info.get('markdown', '')
        headers = table_info.get('headers', [])
        
        if not original_markdown:
            return ""
        
        system_prompt = """You are an expert in Markdown table formatting. Improve the formatting of this table to make it more readable and properly structured.

                        Tasks:
                        1. Ensure proper column alignment
                        2. Improve header clarity if needed
                        3. Format numeric data consistently
                        4. Ensure proper Markdown table syntax
                        5. Make the table more readable

                        Return only the improved Markdown table, nothing else."""
        
        user_content = f"""Improve this table formatting:
                        {original_markdown}
                        Return the improved Markdown table."""
        
        messages = self._create_messages(system_prompt, user_content)
        improved_markdown = self._invoke_llm(messages)
        
        return improved_markdown.strip() if improved_markdown else original_markdown
    
    def _generate_table_summary(self, table_info: Dict[str, Any]) -> str:
        """Generate a summary description of the table"""
        
        headers = table_info.get('headers', [])
        rows = table_info.get('rows', 0)
        columns = table_info.get('columns', 0)
        purpose = table_info.get('table_purpose', '')
        
        system_prompt = """Generate a brief, informative summary of this table that could be used as a caption or description in a document.

                        The summary should:
                        1. Describe what the table shows
                        2. Mention key data types or categories
                        3. Be concise (1-2 sentences)
                        4. Be suitable for document readers

                        Return only the summary text."""
        
        user_content = f"""Generate a summary for this table:

                        Headers: {', '.join(headers)}
                        Dimensions: {rows} rows × {columns} columns
                        Purpose: {purpose}
                        Table content preview: {table_info.get('markdown', '')[:300]}

                        Provide a concise summary."""
        
        messages = self._create_messages(system_prompt, user_content)
        summary = self._invoke_llm(messages)
        
        return summary.strip() if summary else f"Table with {rows} rows and {columns} columns"
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response"""
        
        analysis = {
            'column_types': [],
            'table_purpose': 'data listing',
            'key_insights': [],
            'header_quality': 'fair',
            'suggested_improvements': [],
            'complexity': 'moderate'
        }
        
        try:
            import json
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)
                analysis.update(parsed)
        except Exception as e:
            self.logger.warning(f"Failed to parse table analysis response: {e}")
        
        return analysis


class TablePositionAgent(BaseAgent):
    """Agent specialized in determining table placement in document structure"""
    
    def __init__(self, config):
        super().__init__(
            config=config,
            name="TablePositionAgent",
            role="Determine optimal table placement in Markdown",
            temperature=0.1
        )
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Determine where tables should be placed in the document"""
        
        tables = input_data.get('tables', [])
        text_structure = input_data.get('text_structure', {})
        page_num = input_data.get('page_num', 0)
        
        if not tables:
            return {
                'success': True,
                'result': {'table_placements': []},
                'agent': self.name
            }
        
        # Analyze placement for each table
        placements = []
        for table in tables:
            placement = self._determine_table_placement(table, text_structure)
            placements.append(placement)
        
        return {
            'success': True,
            'result': {'table_placements': placements},
            'agent': self.name
        }
    
    def _determine_table_placement(self, table: Dict[str, Any], text_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal placement for a table"""
        
        complexity = table.get('complexity', 'moderate')
        purpose = table.get('table_purpose', 'data listing')
        rows = table.get('rows', 0)
        columns = table.get('columns', 0)
        
        # Default placement strategy
        placement = {
            'table': table,
            'placement_type': 'block',  # always block for tables
            'position': 'after_paragraph',
            'reference_needed': True,
            'caption': self._generate_table_caption(table),
            'should_export_csv': False
        }
        
        # Determine if table should be exported as CSV
        if rows > 10 or columns > 5 or complexity == 'complex':
            placement['should_export_csv'] = True
            placement['csv_reference'] = f"[Download CSV](tables/{table.get('filename', 'table.csv')})"
        
        # Adjust position based on purpose
        if purpose in ['summary', 'conclusion']:
            placement['position'] = 'end_of_section'
        elif purpose in ['reference', 'appendix']:
            placement['position'] = 'end_of_document'
        
        return placement
    
    def _generate_table_caption(self, table: Dict[str, Any]) -> str:
        """Generate a caption for the table"""
        
        summary = table.get('summary', '')
        page_num = table.get('page_number', 1)
        table_index = table.get('table_index', 0)
        
        # Generate table number
        table_num = f"{page_num}.{table_index + 1}"
        
        if summary:
            return f"Table {table_num}: {summary}"
        else:
            return f"Table {table_num}: Data table with {table.get('rows', 0)} rows and {table.get('columns', 0)} columns"
