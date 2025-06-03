"""
Formula analysis and Markdown conversion agent
"""

# TODO: the temperature variable i think i will put it into the .env file to be more easy to change
# TODO: improve the results on the formula parser, it sucks. I will test if is that 4o-mini is not enough or train an specific model for formulas

from typing import Any, Dict, List, Optional
from pathlib import Path
import re

from .base import BaseAgent
from ..extractors import FormulaExtractor

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os

TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', 0.1))  # Default to 0.1 if not set
class FormulaAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing and converting mathematical formulas to Markdown"""
    
    def __init__(self, config):
        super().__init__(
            config=config,
            name="FormulaAnalysisAgent",
            role="Analyze mathematical formulas and convert to Markdown",
            temperature=TEMPERATURE
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
                
                # Improve LaTeX for Markdown conversion
                improved_latex = self._improve_latex_for_markdown_conversion(formula_info)
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
            r'\\binom', r'\\choose',
            r'[∫∬∭]',  # Unicode integrals
            r'[∑∏]',   # Unicode summation/Product
            r'[∂∇Δ]',  # Unicode partial derivatives, nabla, delta
            r'[√∛∜]',  # Unicode roots
            r'[α-ωΑ-Ω]',  # Greek letters
            r'\d+/\d+',  # Fractions like 1/2
            r'\w+_\w+',  # Subscripts like x_1
            r'\w+\^\w+',  # Superscripts like x^2
            r'lim|sin|cos|tan|log|ln|exp',  # Functions
            r'[≠≤≥≈∞]'  # Math symbols
        ]
        
        return any(re.search(pattern, latex) for pattern in complex_patterns)
    
    def _improve_latex_for_markdown_conversion(self, formula_info: Dict[str, Any]) -> str:
        """Improve LaTeX for Markdown conversion using LLM"""
        
        original_text = formula_info.get('original', '')
        current_latex = formula_info.get('latex', '')
        
        # If we already have good LaTeX, use it
        if current_latex and self._is_valid_latex_for_markdown(current_latex):
            return current_latex
        
        # Only use LLM for complex formulas
        if not self._is_complex_formula(original_text):
            return self._basic_latex_for_markdown_conversion(original_text)
        
        system_prompt = """You are an expert in mathematical notation and LaTeX/Markdown math. Convert this mathematical expression to LaTeX format for use in Markdown.
                        Use standard LaTeX commands (with single backslashes, not double). Return only the LaTeX math expression, without surrounding $ or $$ delimiters."""
        
        user_content = f"""Convert this mathematical expression to LaTeX format for Markdown:
                            Original text: {original_text}
                            Provide clean LaTeX notation using single backslashes (e.g., \\frac, \\sqrt, \\sum)."""
        
        messages = self._create_messages(system_prompt, user_content)
        improved_latex = self._invoke_llm(messages)
        
        # Clean the response
        cleaned_latex = self._clean_latex_response(improved_latex)
        
        return cleaned_latex if cleaned_latex else current_latex
    
    def _basic_latex_for_markdown_conversion(self, text: str) -> str:
        """Convert basic mathematical expressions to LaTeX for Markdown without using LLM."""
        # Common replacements for LaTeX in Markdown
        replacements = {
            '^2': '^{2}',
            '^3': '^{3}',
            '^4': '^{4}',
            '^5': '^{5}',
            '^6': '^{6}',
            '^7': '^{7}',
            '^8': '^{8}',
            '^9': '^{9}',
            '^0': '^{0}',
            '^1': '^{1}',
            'sqrt': r'\sqrt',
            'inf': r'\infty',
            'infinity': r'\infty',
            'alpha': r'\alpha',
            'beta': r'\beta',
            'gamma': r'\gamma',
            'delta': r'\delta',
            'theta': r'\theta',
            'pi': r'\pi',
            'sigma': r'\sigma',
            'omega': r'\omega',
            'lambda': r'\lambda',
            'mu': r'\mu',
            'epsilon': r'\epsilon',
            '+-': r'\pm',
            '<=': r'\leq',
            '>=': r'\geq',
            '!=': r'\neq',
            '~=': r'\approx',
            'sum': r'\sum',
            'integral': r'\int',
            'partial': r'\partial',
            'nabla': r'\nabla',
            'in': r'\in',
            'subset': r'\subset',
            'superset': r'\supset',
            'union': r'\cup',
            'intersection': r'\cap'
        }
        
        latex = text
        for old, new in replacements.items():
            latex = latex.replace(old, new)
        
        return latex
    
    def _is_valid_latex_for_markdown(self, latex: str) -> bool:
        """Check if Markdown math (LaTeX) is valid and well-formed."""
        # Basic validation
        if not latex:
            return False
            
        # Check for common LaTeX/mathematical patterns in Markdown
        valid_patterns = [
            r'\\[a-zA-Z]+',  # LaTeX commands like \frac, \sqrt
            r'\{[^}]*\}',    # Braced groups
            r'\[[^\]]*\]',   # Square brackets
            r'\([^)]*\)',    # Parentheses
            r'[α-ωΑ-Ω]',     # Greek letters
            r'[∫∬∭]',        # Unicode integrals
            r'[∑∏]',         # Unicode summation/Product
            r'[√∛∜]',        # Unicode roots
            r'[²³⁴⁵⁶⁷⁸⁹⁰¹]', # Superscripts
            r'[₀₁₂₃₄₅₆₇₈₉]', # Subscripts
            r'[≠≤≥≈∞±∂∇Δ]',  # Math symbols
            r'[a-zA-Z0-9]',   # Alphanumeric
            r'[+\-*/=(){}[\]]', # Basic operators and brackets
            r'\d+/\d+',       # Fractions
            r'\w+_\w+',       # Subscripts
            r'\w+\^\w+'       # Superscripts
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
        """Clean LLM LaTeX response for Markdown"""
        
        # Remove surrounding delimiters if present
        latex = latex_response.strip()
        
        # Remove $...$ or $$...$$ delimiters if present
        if latex.startswith('$$') and latex.endswith('$$'):
            latex = latex[2:-2]
        elif latex.startswith('$') and latex.endswith('$'):
            latex = latex[1:-1]
        
        # Remove any LaTeX-specific commands that might have leaked in
        latex_commands = [
            r'\\begin\{equation\}', r'\\end\{equation\}',
            r'\\begin\{align\}', r'\\end\{align\}',
            r'\\begin\{math\}', r'\\end\{math\}'
        ]
        
        for cmd in latex_commands:
            latex = re.sub(cmd, '', latex)
        
        # Convert double backslashes to single backslashes for Markdown
        latex = re.sub(r'\\\\([a-zA-Z]+)', r'\\\1', latex)
        
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
    
    def __init__(self, config=None):
        super().__init__(
            config=config,
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
        
        latex_math = formula.get('improved_latex', formula.get('latex', ''))
        formula_type = formula.get('type', 'inline')
        explanation = formula.get('explanation', '')
        
        formatted_formula = formula.copy()
        
        # Format for Markdown with proper delimiters
        if formula_type == 'inline':
            # For inline formulas, wrap with single dollar signs
            formatted_formula['markdown'] = f"${latex_math}$"
        else:
            # Block formula - wrap with double dollar signs and add line breaks
            formatted_formula['markdown'] = f"$$\n{latex_math}\n$$"
            
            # Add explanation if available
            if explanation:
                formatted_formula['markdown'] += f"\n\n*{explanation}*"
        
        # Add alternative text for accessibility
        if explanation:
            formatted_formula['alt_text'] = explanation
        else:
            formatted_formula['alt_text'] = f"Mathematical formula: {formula.get('original', '')}"
        
        return formatted_formula
