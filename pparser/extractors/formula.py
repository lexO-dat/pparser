"""
Mathematical formula detection and conversion
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sympy
from sympy.parsing.latex import parse_latex
from sympy.printing.latex import latex

from .base import BaseExtractor
from ..utils import detect_formula_patterns


class FormulaExtractor(BaseExtractor):
    """Extract and convert mathematical formulas to LaTeX"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.formula_patterns = [
            # LaTeX patterns
            r'\$([^$]+)\$',                    # $formula$
            # r'\\\(([^)]+)\\\)',                # \(formula\)
            # r'\\\[([^\]]+)\\\]',               # \[formula\]
            
            # Mathematical symbols and expressions
            r'[a-zA-Z]\s*[=≈≠<>≤≥]\s*[0-9a-zA-Z\+\-\*/\(\)\^\s]+',
            r'∫[^∫]*d[xyz]',                   # Integrals
            r'∑[^∑]*',                         # Summations
            r'√[^√]*',                         # Square roots
            r'[α-ωΑ-Ω]+',                      # Greek letters
            r'[∂∇∆∞∈∉⊂⊃∪∩]+',                  # Mathematical operators
            
            # Fractions and powers
            r'\d+/\d+',                        # Simple fractions
            r'[a-zA-Z]\^\{?[0-9a-zA-Z]+\}?',   # Powers
            r'[a-zA-Z]_\{?[0-9a-zA-Z]+\}?',   # Subscripts
        ]
        
        # Common mathematical notation mappings
        self.symbol_mappings = {
            '≈': r'\approx',
            '≠': r'\neq',
            '≤': r'\leq',
            '≥': r'\geq',
            '∞': r'\infty',
            '∂': r'\partial',
            '∇': r'\nabla',
            '∆': r'\Delta',
            '∑': r'\sum',
            '∫': r'\int',
            '√': r'\sqrt',
            '±': r'\pm',
            '∈': r'\in',
            '∉': r'\notin',
            '⊂': r'\subset',
            '⊃': r'\supset',
            '∪': r'\cup',
            '∩': r'\cap',
            # Greek letters
            'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta',
            'ε': r'\epsilon', 'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta',
            'ι': r'\iota', 'κ': r'\kappa', 'λ': r'\lambda', 'μ': r'\mu',
            'ν': r'\nu', 'ξ': r'\xi', 'π': r'\pi', 'ρ': r'\rho',
            'σ': r'\sigma', 'τ': r'\tau', 'υ': r'\upsilon', 'φ': r'\phi',
            'χ': r'\chi', 'ψ': r'\psi', 'ω': r'\omega',
            'Α': r'\Alpha', 'Β': r'\Beta', 'Γ': r'\Gamma', 'Δ': r'\Delta',
            'Ε': r'\Epsilon', 'Ζ': r'\Zeta', 'Η': r'\Eta', 'Θ': r'\Theta',
            'Ι': r'\Iota', 'Κ': r'\Kappa', 'Λ': r'\Lambda', 'Μ': r'\Mu',
            'Ν': r'\Nu', 'Ξ': r'\Xi', 'Π': r'\Pi', 'Ρ': r'\Rho',
            'Σ': r'\Sigma', 'Τ': r'\Tau', 'Υ': r'\Upsilon', 'Φ': r'\Phi',
            'Χ': r'\Chi', 'Ψ': r'\Psi', 'Ω': r'\Omega',
        }
    
    def extract(self, pdf_path: Path, page_num: int, **kwargs) -> Dict[str, Any]:
        """Extract formulas from a specific page"""
        
        result = {
            'type': 'formulas',
            'formulas': [],
            'inline_formulas': [],
            'block_formulas': [],
            'total_formulas': 0
        }
        
        try:
            # Get text from the page (using text extractor)
            from .text import TextExtractor
            text_extractor = TextExtractor()
            text_result = text_extractor.extract(pdf_path, page_num)
            text_content = text_result.get('content', '')
            
            if not text_content:
                return result
            
            # Extract formulas using different methods
            all_formulas = []
            
            # Pattern-based extraction
            pattern_formulas = self._extract_with_patterns(text_content)
            all_formulas.extend(pattern_formulas)

            # Context-based extraction
            context_formulas = self._extract_with_context(text_content)
            all_formulas.extend(context_formulas)

            # Symbol density analysis
            density_formulas = self._extract_with_symbol_density(text_content)
            all_formulas.extend(density_formulas)
            
            # Process and classify formulas
            processed_formulas = self._process_formulas(all_formulas)
            
            # Categorize formulas
            for formula in processed_formulas:
                if formula['type'] == 'inline':
                    result['inline_formulas'].append(formula)
                else:
                    result['block_formulas'].append(formula)
                
                result['formulas'].append(formula)
            
            result['total_formulas'] = len(result['formulas'])
            
        except Exception as e:
            self.logger.error(f"Error extracting formulas from page {page_num + 1}: {e}")
        
        return result
    
    def _extract_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract formulas using regex patterns"""
        formulas = []
        
        for pattern in self.formula_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                formula_text = match.group(0)
                if self._is_likely_formula(formula_text):
                    formulas.append({
                        'text': formula_text,
                        'start': match.start(),
                        'end': match.end(),
                        'method': 'pattern'
                    })
        
        return formulas
    
    def _extract_with_context(self, text: str) -> List[Dict[str, Any]]:
        """Extract formulas by analyzing context"""
        formulas = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for lines that are likely to contain formulas
            if self._has_formula_context(line, lines, i):
                # Extract potential formulas from the line
                potential_formulas = self._find_formulas_in_line(line)
                for formula in potential_formulas:
                    formulas.append({
                        'text': formula,
                        'line': i + 1,
                        'context': line,
                        'method': 'context'
                    })
        
        return formulas
    
    def _extract_with_symbol_density(self, text: str) -> List[Dict[str, Any]]:
        """Extract formulas by analyzing mathematical symbol density"""
        formulas = []
        
        # Split text into chunks
        chunks = re.split(r'[.!?]\s+', text)
        
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 5:
                continue
            
            # Calculate mathematical symbol density
            math_symbols = sum(1 for char in chunk if char in '∫∑√±∞∂∇∆≈≠≤≥∈∉⊂⊃∪∩αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ()[]{}^_=+-*/')
            density = math_symbols / len(chunk)
            
            if density > 0.1:  # Threshold for mathematical content
                formulas.append({
                    'text': chunk,
                    'density': density,
                    'method': 'density'
                })
        
        return formulas
    
    def _is_likely_formula(self, text: str) -> bool:
        """Check if text is likely to be a mathematical formula"""
        
        # Too short or too long
        if len(text) < 2 or len(text) > 200:
            return False
        
        # Contains mathematical symbols
        math_symbols = '∫∑√±∞∂∇∆≈≠≤≥∈∉⊂⊃∪∩^_=+-*/()'
        if any(sym in text for sym in math_symbols):
            return True
        
        # Contains Greek letters
        greek_pattern = r'[α-ωΑ-Ω]'
        if re.search(greek_pattern, text):
            return True
        
        # Contains fractions or powers
        if re.search(r'\d+/\d+|[a-zA-Z]\^[0-9a-zA-Z]|[a-zA-Z]_[0-9a-zA-Z]', text):
            return True
        
        # Equation-like structure
        if re.search(r'[a-zA-Z]\s*[=≈]\s*[0-9a-zA-Z\+\-\*/\(\)]+', text):
            return True
        
        return False
    
    def _has_formula_context(self, line: str, lines: List[str], line_index: int) -> bool:
        """Check if line has context suggesting mathematical content"""
        
        # Keywords that often appear near formulas
        formula_keywords = [
            'equation', 'formula', 'calculate', 'solve', 'derivative',
            'integral', 'function', 'where', 'given', 'let', 'theorem',
            'proof', 'therefore', 'thus', 'hence', 'if and only if'
        ]
        
        # Check current line and surrounding lines
        context_lines = []
        for i in range(max(0, line_index - 1), min(len(lines), line_index + 2)):
            context_lines.append(lines[i].lower())
        
        context_text = ' '.join(context_lines)
        
        return any(keyword in context_text for keyword in formula_keywords)
    
    def _find_formulas_in_line(self, line: str) -> List[str]:
        """Find potential formulas within a line"""
        formulas = []
        
        # Look for mathematical expressions
        expressions = re.findall(r'[a-zA-Z0-9\s]*[=≈≠<>≤≥][a-zA-Z0-9\s\+\-\*/\(\)\^_]+', line)
        for expr in expressions:
            if self._is_likely_formula(expr):
                formulas.append(expr.strip())
        
        return formulas
    
    def _process_formulas(self, raw_formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and convert formulas to LaTeX"""
        processed = []
        seen = set()
        
        for formula_data in raw_formulas:
            formula_text = formula_data['text'].strip()
            
            # Skip duplicates
            if formula_text in seen:
                continue
            seen.add(formula_text)
            
            # Convert to LaTeX
            latex_formula = self._convert_to_latex(formula_text)
            
            # Determine if inline or block formula
            formula_type = self._determine_formula_type(formula_text, formula_data)
            
            processed_formula = {
                'original': formula_text,
                'latex': latex_formula,
                'type': formula_type,
                'method': formula_data.get('method', 'unknown'),
                'confidence': self._calculate_confidence(formula_text),
                'description': self._generate_description(latex_formula)
            }
            
            # Add position information if available
            if 'start' in formula_data:
                processed_formula['position'] = {
                    'start': formula_data['start'],
                    'end': formula_data['end']
                }
            
            if 'line' in formula_data:
                processed_formula['line'] = formula_data['line']
            
            processed.append(processed_formula)
        
        return processed
    
    def _convert_to_latex(self, formula_text: str) -> str:
        """Convert formula text to LaTeX format"""
        
        # Start with the original text
        latex_text = formula_text
        
        # Apply symbol mappings
        for symbol, latex_symbol in self.symbol_mappings.items():
            latex_text = latex_text.replace(symbol, latex_symbol)
        
        # Handle special cases
        latex_text = self._handle_special_conversions(latex_text)
        
        # Try to parse and improve with SymPy
        try:
            # Clean for SymPy parsing
            cleaned = self._clean_for_sympy(latex_text)
            if cleaned:
                # Try to parse as LaTeX first
                try:
                    expr = parse_latex(cleaned)
                    latex_text = latex(expr)
                except:
                    # Try to parse as regular expression
                    try:
                        expr = sympy.sympify(cleaned, evaluate=False)
                        latex_text = latex(expr)
                    except:
                        pass  # Keep original conversion
        except:
            pass  # Keep manual conversion
        
        return latex_text
    
    def _handle_special_conversions(self, text: str) -> str:
        """Handle special conversion cases"""
        
        # Convert fractions
        text = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', text)
        
        # Convert powers
        text = re.sub(r'([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)', r'\1^{\2}', text)
        text = re.sub(r'([a-zA-Z0-9]+)\^{([^}]+)}', r'\1^{\2}', text)
        
        # Convert subscripts
        text = re.sub(r'([a-zA-Z0-9]+)_([a-zA-Z0-9]+)', r'\1_{\2}', text)
        text = re.sub(r'([a-zA-Z0-9]+)_{([^}]+)}', r'\1_{\2}', text)
        
        # Convert square roots
        text = re.sub(r'√\(([^)]+)\)', r'\\sqrt{\1}', text)
        text = re.sub(r'√([a-zA-Z0-9]+)', r'\\sqrt{\1}', text)
        
        return text
    
    def _clean_for_sympy(self, text: str) -> str:
        """Clean text for SymPy parsing"""
        
        # Remove LaTeX commands that SymPy might not understand
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text if text else None
    
    def _determine_formula_type(self, formula_text: str, formula_data: Dict[str, Any]) -> str:
        """Determine if formula should be inline or block"""
        
        # Block formulas are typically:
        # - Longer formulas
        # - Formulas on their own line
        # - Complex expressions with multiple operators
        
        if len(formula_text) > 50:
            return 'block'
        
        # Check if formula contains complex elements
        complex_elements = ['∫', '∑', '\\frac', '\\sqrt', '\\begin', '\\end']
        if any(elem in formula_text for elem in complex_elements):
            return 'block'
        
        # Check context
        if formula_data.get('context', '').strip() == formula_text.strip():
            return 'block'  # Formula is on its own line
        
        return 'inline'
    
    def _calculate_confidence(self, formula_text: str) -> float:
        """Calculate confidence score for formula detection"""
        
        score = 0.0
        
        # Mathematical symbols increase confidence
        math_symbols = '∫∑√±∞∂∇∆≈≠≤≥∈∉⊂⊃∪∩^_=+-*/()'
        symbol_count = sum(1 for char in formula_text if char in math_symbols)
        score += min(symbol_count * 0.1, 0.5)
        
        # Greek letters increase confidence
        greek_pattern = r'[α-ωΑ-Ω]'
        if re.search(greek_pattern, formula_text):
            score += 0.2
        
        # Equation structure increases confidence
        if re.search(r'[a-zA-Z]\s*[=≈]\s*[0-9a-zA-Z\+\-\*/\(\)]+', formula_text):
            score += 0.3
        
        # LaTeX formatting increases confidence
        if '\\' in formula_text or '$' in formula_text:
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_description(self, latex_formula: str) -> str:
        """Generate a human-readable description of the formula"""
        
        # TODO: create better NLP techni1ques for this part
        
        descriptions = []
        
        if '\frac' in latex_formula:
            descriptions.append("fraction")
        
        if '\sqrt' in latex_formula:
            descriptions.append("square root")
        
        if '\int' in latex_formula:
            descriptions.append("integral")
        
        if '\sum' in latex_formula:
            descriptions.append("summation")
        
        if '^' in latex_formula:
            descriptions.append("exponential")
        
        if '_' in latex_formula:
            descriptions.append("subscript")
        
        if '=' in latex_formula:
            descriptions.append("equation")
        
        if descriptions:
            return "Mathematical expression with " + ", ".join(descriptions)
        else:
            return "Mathematical expression"
