"""
Formula analysis and LaTeX conversion agent
"""

# TODO: the temperature variable i think i will put it into the .env file to be more easy to change
# TODO: improve the results on the formula parser, it sucks. I will test if is that 4o-mini is not enough or train an specific model for formulas

from typing import Any, Dict, List, Optional
from pathlib import Path
import re

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
        batch_size = 5  # Process formulas in smaller batches
        
        formulas = extraction_result.get('formulas', [])
        for i in range(0, len(formulas), batch_size):
            batch = formulas[i:i + batch_size]
            
            for formula_info in batch:
                enhanced_formula = formula_info.copy()
                
                # Improve LaTeX conversion
                improved_latex = self._improve_latex_conversion(formula_info)
                enhanced_formula['improved_latex'] = improved_latex
                
                # Only analyze complex formulas
                if self._is_complex_formula(improved_latex):
                    analysis = self._analyze_mathematical_content(formula_info, improved_latex)
                    enhanced_formula.update(analysis)
                    
                    # Generate explanation only for complex formulas
                    explanation = self._generate_formula_explanation(enhanced_formula)
                    enhanced_formula['explanation'] = explanation
                else:
                    # Simple formula handling
                    enhanced_formula.update({
                        'math_type': 'basic',
                        'complexity': 'elementary',
                        'explanation': f"Basic mathematical expression: {formula_info.get('original', '')}"
                    })
                
                enhanced_formulas.append(enhanced_formula)
        
        # Update categorization
        result = extraction_result.copy()
        result['formulas'] = enhanced_formulas
        result['inline_formulas'] = [f for f in enhanced_formulas if f.get('type') == 'inline']
        result['block_formulas'] = [f for f in enhanced_formulas if f.get('type') == 'block']
        
        return result
    
    def _is_complex_formula(self, latex: str) -> bool:
        """Determine if a formula is complex enough to warrant detailed analysis."""
        complex_patterns = [
            r'\\frac', r'\\sqrt', r'\\int', r'\\sum', r'\\prod',
            r'\\lim', r'\\partial', r'\\nabla', r'\\Delta',
            r'\\begin\{.*\}', r'\\end\{.*\}',
            r'\\left', r'\\right',
            r'\\over', r'\\atop',
            r'\\binom', r'\\choose'
        ]
        
        return any(re.search(pattern, latex) for pattern in complex_patterns)
    
    def _improve_latex_conversion(self, formula_info: Dict[str, Any]) -> str:
        """Improve LaTeX conversion using LLM"""
        
        original_text = formula_info.get('original', '')
        current_latex = formula_info.get('latex', '')
        
        # If we already have good LaTeX, use it
        if current_latex and self._is_valid_latex(current_latex):
            return current_latex
        
        # Only use LLM for complex formulas
        if not self._is_complex_formula(original_text):
            return self._basic_latex_conversion(original_text)
        
        system_prompt = """You are an expert in mathematical notation and LaTeX. Convert this mathematical expression to LaTeX.

Return only the LaTeX code, without surrounding $ or $$ delimiters."""
        
        user_content = f"""Convert this mathematical expression to LaTeX:

Original text: {original_text}

Provide LaTeX code."""
        
        messages = self._create_messages(system_prompt, user_content)
        improved_latex = self._invoke_llm(messages)
        
        # Clean the response
        cleaned_latex = self._clean_latex_response(improved_latex)
        
        return cleaned_latex if cleaned_latex else current_latex
    
    def _basic_latex_conversion(self, text: str) -> str:
        """Convert basic mathematical expressions to LaTeX without using LLM."""
        # Common replacements
        replacements = {
            '^2': '^{2}',
            '^3': '^{3}',
            'sqrt': '\\sqrt',
            'inf': '\\infty',
            'alpha': '\\alpha',
            'beta': '\\beta',
            'gamma': '\\gamma',
            'delta': '\\delta',
            'theta': '\\theta',
            'pi': '\\pi',
            'sigma': '\\sigma',
            'omega': '\\omega'
        }
        
        latex = text
        for old, new in replacements.items():
            latex = latex.replace(old, new)
        
        return latex
    
    def _is_valid_latex(self, latex: str) -> bool:
        """Check if LaTeX is valid and well-formed."""
        # Basic validation
        if not latex:
            return False
            
        # Check for common LaTeX patterns
        valid_patterns = [
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'\{[^}]*\}',    # Braced groups
            r'\[[^\]]*\]',   # Square brackets
            r'\([^)]*\)',    # Parentheses
            r'[a-zA-Z0-9]',  # Alphanumeric
            r'[+\-*/=]',     # Basic operators
            r'[α-ωΑ-Ω]'      # Greek letters
        ]
        
        return any(re.search(pattern, latex) for pattern in valid_patterns)
    
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
