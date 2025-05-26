"""
Formula analysis and LaTeX conversion agent
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from .base import BaseAgent
from ..extractors import FormulaExtractor


class FormulaAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing and converting mathematical formulas"""
    
    def __init__(self, config):
        super().__init__(
            config=config,
            name="FormulaAnalysisAgent",
            role="Analyze mathematical formulas and convert to LaTeX",
            temperature=0.1
        )
        self.extractor = FormulaExtractor()
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process PDF page and analyze formulas"""
        
        pdf_path = Path(input_data.get('pdf_path'))
        page_num = input_data.get('page_num', 0)
        
        # Extract formulas
        extraction_result = self.extractor.extract(pdf_path, page_num)
        
        if not extraction_result.get('formulas'):
            return {
                'success': True,
                'result': extraction_result,
                'agent': self.name,
                'message': 'No formulas found on this page'
            }
        
        # Enhance formula analysis with LLM
        enhanced_result = self._enhance_formula_analysis(extraction_result)
        
        return {
            'success': True,
            'result': enhanced_result,
            'agent': self.name
        }
    
    def _enhance_formula_analysis(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance formula analysis using LLM"""
        
        enhanced_formulas = []
        
        for formula_info in extraction_result.get('formulas', []):
            enhanced_formula = formula_info.copy()
            
            # Improve LaTeX conversion
            improved_latex = self._improve_latex_conversion(formula_info)
            enhanced_formula['improved_latex'] = improved_latex
            
            # Analyze mathematical content
            analysis = self._analyze_mathematical_content(formula_info, improved_latex)
            enhanced_formula.update(analysis)
            
            # Generate explanation
            explanation = self._generate_formula_explanation(enhanced_formula)
            enhanced_formula['explanation'] = explanation
            
            enhanced_formulas.append(enhanced_formula)
        
        # Update categorization
        result = extraction_result.copy()
        result['formulas'] = enhanced_formulas
        result['inline_formulas'] = [f for f in enhanced_formulas if f.get('type') == 'inline']
        result['block_formulas'] = [f for f in enhanced_formulas if f.get('type') == 'block']
        
        return result
    
    def _improve_latex_conversion(self, formula_info: Dict[str, Any]) -> str:
        """Improve LaTeX conversion using LLM"""
        
        original_text = formula_info.get('original', '')
        current_latex = formula_info.get('latex', '')
        
        system_prompt = """You are an expert in mathematical notation and LaTeX. Improve the LaTeX representation of this mathematical expression to ensure it's accurate and well-formatted.

Tasks:
1. Correct any LaTeX syntax errors
2. Improve mathematical notation for clarity
3. Use appropriate LaTeX commands for mathematical symbols
4. Ensure proper grouping with braces
5. Follow standard mathematical typesetting conventions

Return only the improved LaTeX code, without surrounding $ or $$ delimiters."""
        
        user_content = f"""Improve the LaTeX for this mathematical expression:

Original text: {original_text}
Current LaTeX: {current_latex}

Provide improved LaTeX code."""
        
        messages = self._create_messages(system_prompt, user_content)
        improved_latex = self._invoke_llm(messages)
        
        # Clean the response
        cleaned_latex = self._clean_latex_response(improved_latex)
        
        return cleaned_latex if cleaned_latex else current_latex
    
    def _analyze_mathematical_content(self, formula_info: Dict[str, Any], latex: str) -> Dict[str, Any]:
        """Analyze the mathematical content of the formula"""
        
        system_prompt = """You are a mathematics expert. Analyze this mathematical expression and classify it.

Provide analysis in JSON format:
{
    "math_type": "algebra|calculus|geometry|statistics|logic|other",
    "complexity": "elementary|intermediate|advanced",
    "variables": ["x", "y", "z"],
    "operators": ["integral", "derivative", "summation"],
    "concepts": ["concept1", "concept2"],
    "field": "mathematics|physics|engineering|economics|other"
}"""
        
        original = formula_info.get('original', '')
        
        user_content = f"""Analyze this mathematical expression:

Original: {original}
LaTeX: {latex}

Provide mathematical analysis."""
        
        messages = self._create_messages(system_prompt, user_content)
        llm_response = self._invoke_llm(messages)
        
        # Parse analysis response
        analysis = self._parse_math_analysis(llm_response)
        
        return analysis
    
    def _generate_formula_explanation(self, formula_info: Dict[str, Any]) -> str:
        """Generate an explanation of what the formula represents"""
        
        latex = formula_info.get('improved_latex', formula_info.get('latex', ''))
        math_type = formula_info.get('math_type', 'unknown')
        complexity = formula_info.get('complexity', 'intermediate')
        
        system_prompt = """Generate a brief, clear explanation of what this mathematical formula represents. The explanation should be accessible to someone with basic mathematical knowledge.

Keep it concise (1-2 sentences) and focus on what the formula does or represents, not how to solve it."""
        
        user_content = f"""Explain this mathematical formula:

LaTeX: {latex}
Type: {math_type}
Complexity: {complexity}

Provide a clear explanation."""
        
        messages = self._create_messages(system_prompt, user_content)
        explanation = self._invoke_llm(messages)
        
        return explanation.strip() if explanation else "Mathematical expression"
    
    def _clean_latex_response(self, latex_response: str) -> str:
        """Clean LLM LaTeX response"""
        
        # Remove surrounding delimiters if present
        latex = latex_response.strip()
        
        # Remove $...$ or $$...$$ delimiters
        if latex.startswith('$$') and latex.endswith('$$'):
            latex = latex[2:-2]
        elif latex.startswith('$') and latex.endswith('$'):
            latex = latex[1:-1]
        
        # Remove \begin{equation} and \end{equation} if present
        if '\\begin{equation}' in latex:
            latex = latex.replace('\\begin{equation}', '').replace('\\end{equation}', '')
        
        return latex.strip()
    
    def _parse_math_analysis(self, response: str) -> Dict[str, Any]:
        """Parse mathematical analysis response"""
        
        analysis = {
            'math_type': 'other',
            'complexity': 'intermediate',
            'variables': [],
            'operators': [],
            'concepts': [],
            'field': 'mathematics'
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
            self.logger.warning(f"Failed to parse math analysis response: {e}")
        
        return analysis


class FormulaFormattingAgent(BaseAgent):
    """Agent specialized in formatting formulas for Markdown"""
    
    def __init__(self):
        super().__init__(
            name="FormulaFormattingAgent",
            role="Format mathematical formulas for Markdown output",
            temperature=0.0
        )
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Format formulas for Markdown output"""
        
        formulas = input_data.get('formulas', [])
        
        if not formulas:
            return {
                'success': True,
                'result': {'formatted_formulas': []},
                'agent': self.name
            }
        
        # Format each formula for Markdown
        formatted_formulas = []
        
        for formula in formulas:
            formatted = self._format_formula_for_markdown(formula)
            formatted_formulas.append(formatted)
        
        return {
            'success': True,
            'result': {'formatted_formulas': formatted_formulas},
            'agent': self.name
        }
    
    def _format_formula_for_markdown(self, formula: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single formula for Markdown"""
        
        latex = formula.get('improved_latex', formula.get('latex', ''))
        formula_type = formula.get('type', 'inline')
        explanation = formula.get('explanation', '')
        
        formatted_formula = formula.copy()
        
        # Format for Markdown
        if formula_type == 'inline':
            formatted_formula['markdown'] = f"${latex}$"
        else:
            # Block formula
            formatted_formula['markdown'] = f"$$\n{latex}\n$$"
            
            # Add explanation if available
            if explanation:
                formatted_formula['markdown'] += f"\n\n*{explanation}*"
        
        # Add alternative text for accessibility
        if explanation:
            formatted_formula['alt_text'] = explanation
        else:
            formatted_formula['alt_text'] = f"Mathematical formula: {formula.get('original', '')}"
        
        return formatted_formula
