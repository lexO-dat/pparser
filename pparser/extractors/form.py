"""
Form and survey detection
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseExtractor


class FormExtractor(BaseExtractor):
    """Extract forms and survey questions from PDF pages"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Patterns for different question types
        self.question_patterns = [
            # Multiple choice patterns
            r'(?i)(?:pregunta|question|q\d+)[:\.]?\s*(.+?)(?=\n\s*[a-e]\)|$)',
            r'(?i)\d+\.\s*(.+?)(?=\n\s*[a-e]\)|$)',
            
            # True/False patterns
            r'(?i)(.+?)\s*\(?\s*(?:verdadero|falso|true|false|v|f)\s*\)?',
            
            # Fill-in-the-blank patterns
            r'(.+?)_+(.+?)',
            r'(.+?)\[.*?\](.+?)',
            
            # Rating scale patterns
            r'(.+?)\s*\d+\s*-\s*\d+',
        ]
        
        # Patterns for answer options
        self.option_patterns = [
            r'^\s*([a-e])\)\s*(.+)$',           # a) option
            r'^\s*([a-e])\.\s*(.+)$',           # a. option
            r'^\s*\(?([a-e])\)\s*(.+)$',        # (a) option
            r'^\s*([a-e]):\s*(.+)$',            # a: option
            r'^\s*-\s*(.+)$',                   # - option
            r'^\s*•\s*(.+)$',                   # • option
            r'^\s*\*\s*(.+)$',                  # * option
        ]
        
        # Keywords that indicate questions
        self.question_keywords = [
            'pregunta', 'question', 'cuál', 'which', 'what', 'how', 'why',
            'when', 'where', 'who', 'qué', 'cómo', 'por qué', 'cuándo',
            'dónde', 'quién', 'seleccione', 'select', 'choose', 'mark',
            'indicate', 'complete', 'fill'
        ]
    
    def extract(self, pdf_path: Path, page_num: int, **kwargs) -> Dict[str, Any]:
        """Extract forms and surveys from a specific page"""
        
        result = {
            'type': 'forms',
            'questions': [],
            'forms': [],
            'total_questions': 0,
            'total_forms': 0
        }
        
        try:
            # Get text from the page
            from .text import TextExtractor
            text_extractor = TextExtractor()
            text_result = text_extractor.extract(pdf_path, page_num)
            text_content = text_result.get('content', '')
            
            if not text_content:
                return result
            
            # Extract questions and forms
            questions = self._extract_questions(text_content)
            forms = self._extract_forms(text_content)
            
            result['questions'] = questions
            result['forms'] = forms
            result['total_questions'] = len(questions)
            result['total_forms'] = len(forms)
            
        except Exception as e:
            self.logger.error(f"Error extracting forms from page {page_num + 1}: {e}")
        
        return result
    
    def _extract_questions(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual questions from text"""
        questions = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if self._is_question_line(line):
                question_data = self._parse_question(lines, i)
                if question_data:
                    questions.append(question_data)
                    i = question_data.get('end_line', i + 1)
                else:
                    i += 1
            else:
                i += 1
        
        return questions
    
    def _extract_forms(self, text: str) -> List[Dict[str, Any]]:
        """Extract complete forms from text"""
        forms = []
        
        # Look for form patterns
        form_indicators = [
            'formulario', 'form', 'encuesta', 'survey', 'questionnaire',
            'cuestionario', 'evaluación', 'evaluation', 'examen', 'exam'
        ]
        
        lines = text.split('\n')
        current_form = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if line indicates start of a form
            if any(indicator in line_lower for indicator in form_indicators):
                if current_form:
                    forms.append(current_form)
                
                current_form = {
                    'title': line.strip(),
                    'start_line': i + 1,
                    'questions': [],
                    'instructions': []
                }
            
            elif current_form:
                # Add content to current form
                if self._is_question_line(line):
                    question_data = self._parse_question(lines, i)
                    if question_data:
                        current_form['questions'].append(question_data)
                elif self._is_instruction_line(line):
                    current_form['instructions'].append(line.strip())
        
        # Add final form
        if current_form:
            forms.append(current_form)
        
        return forms
    
    def _is_question_line(self, line: str) -> bool:
        """Check if a line contains a question"""
        line_lower = line.lower().strip()
        
        # Check for question keywords
        if any(keyword in line_lower for keyword in self.question_keywords):
            return True
        
        # Check for question patterns
        for pattern in self.question_patterns:
            if re.search(pattern, line):
                return True
        
        # Check for numbered questions
        if re.match(r'^\s*\d+\.\s*[A-Z]', line):
            return True
        
        # Check for question marks
        if '?' in line:
            return True
        
        return False
    
    def _is_instruction_line(self, line: str) -> bool:
        """Check if a line contains instructions"""
        instruction_keywords = [
            'instrucciones', 'instructions', 'indicaciones', 'nota',
            'note', 'importante', 'important', 'atención', 'attention'
        ]
        
        line_lower = line.lower().strip()
        return any(keyword in line_lower for keyword in instruction_keywords)
    
    def _parse_question(self, lines: List[str], start_index: int) -> Optional[Dict[str, Any]]:
        """Parse a question and its options"""
        
        question_line = lines[start_index].strip()
        question_data = {
            'question': question_line,
            'type': 'unknown',
            'options': [],
            'start_line': start_index + 1,
            'end_line': start_index + 1
        }
        
        # Look for options in following lines
        i = start_index + 1
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check if this line is an option
            option_data = self._parse_option(line)
            if option_data:
                question_data['options'].append(option_data)
                question_data['end_line'] = i + 1
                i += 1
            else:
                # Check if this might be the start of another question
                if self._is_question_line(line):
                    break
                
                # Check if this is continuation of the question
                if i == start_index + 1 and not any(char in line for char in 'abcde)'):
                    question_data['question'] += ' ' + line
                    question_data['end_line'] = i + 1
                
                i += 1
        
        # Determine question type
        question_data['type'] = self._determine_question_type(question_data)
        
        # Generate markdown representation
        question_data['markdown'] = self._question_to_markdown(question_data)
        
        return question_data if question_data['question'] else None
    
    def _parse_option(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse an answer option"""
        
        for pattern in self.option_patterns:
            match = re.match(pattern, line)
            if match:
                if len(match.groups()) == 2:
                    # Pattern with option letter/marker
                    return {
                        'marker': match.group(1),
                        'text': match.group(2).strip(),
                        'selected': False
                    }
                else:
                    # Pattern without explicit marker
                    return {
                        'marker': '-',
                        'text': match.group(1).strip(),
                        'selected': False
                    }
        
        return None
    
    def _determine_question_type(self, question_data: Dict[str, Any]) -> str:
        """Determine the type of question"""
        
        question_text = question_data['question'].lower()
        options = question_data['options']
        
        # Multiple choice
        if len(options) >= 2:
            # Check for true/false
            if len(options) == 2:
                option_texts = [opt['text'].lower() for opt in options]
                tf_keywords = ['true', 'false', 'verdadero', 'falso', 'v', 'f']
                if any(keyword in ' '.join(option_texts) for keyword in tf_keywords):
                    return 'true_false'
            
            return 'multiple_choice'
        
        # Fill in the blank
        if '_' in question_text or '[' in question_text:
            return 'fill_blank'
        
        # Rating scale
        if re.search(r'\d+\s*-\s*\d+', question_text):
            return 'rating_scale'
        
        # Short answer (has question mark but no options)
        if '?' in question_text:
            return 'short_answer'
        
        return 'unknown'
    
    def _question_to_markdown(self, question_data: Dict[str, Any]) -> str:
        """Convert question to Markdown format"""
        
        lines = []
        
        # Question text
        lines.append(f"**Pregunta:** {question_data['question']}")
        
        # Options
        if question_data['options']:
            for option in question_data['options']:
                checkbox = '[ ]'  # Default to unchecked
                if option.get('selected', False):
                    checkbox = '[x]'
                
                marker = option.get('marker', '-')
                text = option.get('text', '')
                
                if marker in 'abcde':
                    lines.append(f"- {checkbox} {marker.upper()}) {text}")
                else:
                    lines.append(f"- {checkbox} {text}")
        
        return '\n'.join(lines)
    
    def detect_filled_forms(self, pdf_path: Path, page_num: int) -> Dict[str, Any]:
        """Detect forms that might already be filled"""
        
        result = {
            'type': 'filled_forms',
            'detected_marks': [],
            'checkboxes': [],
            'filled_fields': []
        }
        
        try:
           # TODO: implement better image procesing, now, this is trash :(
            
            from .text import TextExtractor
            text_extractor = TextExtractor()
            text_result = text_extractor.extract(pdf_path, page_num)
            text_content = text_result.get('content', '')
            
            # Look for patterns that suggest filled forms
            filled_patterns = [
                r'[x✓✗]\s*[a-e]\)',  # Marked options
                r'\[[x✓✗]\]',         # Checked boxes
                r'___[^_]+___',        # Filled blanks
            ]
            
            for pattern in filled_patterns:
                matches = re.findall(pattern, text_content)
                result['detected_marks'].extend(matches)
            
        except Exception as e:
            self.logger.error(f"Error detecting filled forms: {e}")
        
        return result
