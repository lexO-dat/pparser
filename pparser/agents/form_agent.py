"""
Form and survey analysis agent
"""

# TODO: the temperature variable i think i will put it into the .env file to be more easy to change

from typing import Any, Dict, List, Optional
from pathlib import Path

from .base import BaseAgent
from ..extractors import FormExtractor

from dotenv import load_dotenv
import os

# see if the env file was successfully loaded
if not load_dotenv():
    print("Warning: .env file not found or could not be loaded. Default values will be used.")

TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))  # Default temperature if not set in .env

class FormAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing forms and survey questions"""
    
    def __init__(self, config):
        super().__init__(
            config=config,
            name="FormAnalysisAgent",
            role="Analyze forms and convert to interactive Markdown",
            temperature=TEMPERATURE
        )
        self.extractor = FormExtractor()
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process PDF page and analyze forms"""
        
        pdf_path = Path(input_data.get('pdf_path'))
        page_num = input_data.get('page_num', 0)
        
        # Extract forms and questions
        extraction_result = self.extractor.extract(pdf_path, page_num)
        
        if not extraction_result.get('questions') and not extraction_result.get('forms'):
            return {
                'success': True,
                'result': extraction_result,
                'agent': self.name,
                'message': 'No forms or questions found on this page'
            }
        
        # Enhance form analysis with LLM
        enhanced_result = self._enhance_form_analysis(extraction_result)
        
        return {
            'success': True,
            'result': enhanced_result,
            'agent': self.name
        }
    
    def _enhance_form_analysis(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance form analysis using LLM"""
        
        # Enhance individual questions
        enhanced_questions = []
        for question in extraction_result.get('questions', []):
            enhanced_question = self._enhance_question(question)
            enhanced_questions.append(enhanced_question)
        
        # Enhance complete forms
        enhanced_forms = []
        for form in extraction_result.get('forms', []):
            enhanced_form = self._enhance_form(form)
            enhanced_forms.append(enhanced_form)
        
        result = extraction_result.copy()
        result['questions'] = enhanced_questions
        result['forms'] = enhanced_forms
        
        return result
    
    def _enhance_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a single question with LLM analysis"""
        
        question_text = question.get('question', '')
        options = question.get('options', [])
        current_type = question.get('type', 'unknown')
        
        # Improve question classification
        improved_type = self._classify_question_type(question_text, options)
        
        # Improve question formatting
        improved_markdown = self._improve_question_formatting(question)
        
        # Add metadata
        metadata = self._analyze_question_metadata(question_text, options)
        
        enhanced = question.copy()
        enhanced['improved_type'] = improved_type
        enhanced['improved_markdown'] = improved_markdown
        enhanced.update(metadata)
        
        return enhanced
    
    def _enhance_form(self, form: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a complete form with LLM analysis"""
        
        # TODO: remember wtf i created this variables :)
        title = form.get('title', '')
        questions = form.get('questions', [])
        instructions = form.get('instructions', [])
        
        # Analyze form structure and purpose
        analysis = self._analyze_form_structure(form)
        
        # Generate form metadata
        metadata = self._generate_form_metadata(form)
        
        # Create complete form markdown
        complete_markdown = self._create_complete_form_markdown(form, analysis)
        
        enhanced = form.copy()
        enhanced.update(analysis)
        enhanced.update(metadata)
        enhanced['complete_markdown'] = complete_markdown
        
        return enhanced
    
    def _classify_question_type(self, question_text: str, options: List[Dict[str, Any]]) -> str:
        """Classify question type using LLM"""
        
        system_prompt = """
                            Classify this question type based on the question text and options provided.

                            Question types:
                            - multiple_choice: Select one from several options
                            - multiple_select: Select multiple from several options  
                            - true_false: Binary true/false question
                            - likert_scale: Rating scale (1-5, strongly agree to disagree, etc.)
                            - fill_blank: Fill in missing information
                            - short_answer: Open-ended short response
                            - long_answer: Open-ended long response
                            - ranking: Rank items in order
                            - matching: Match items between lists

                            Return only the question type, nothing else.
                        """
        
        options_text = '\n'.join([f"- {opt.get('text', '')}" for opt in options])
        
        user_content = f"""
                            Question: {question_text}

                            Options:  {options_text}

                            Classify the question type.
                        """
        
        messages = self._create_messages(system_prompt, user_content)
        question_type = self._invoke_llm(messages)
        
        return question_type.strip().lower() if question_type else 'unknown'
    
    def _improve_question_formatting(self, question: Dict[str, Any]) -> str:
        """Improve question formatting for Markdown"""
        
        question_text = question.get('question', '')
        options = question.get('options', [])
        question_type = question.get('improved_type', question.get('type', 'unknown'))
        
        system_prompt = """
                            Format this question as clean, interactive Markdown with proper checkboxes or formatting based on the question type.

                            Guidelines:
                            - Use **bold** for question text
                            - Use proper checkbox syntax: - [ ] for unchecked, - [x] for checked
                            - Format options clearly and consistently
                            - Add appropriate spacing and structure
                            - For rating scales, use a clear scale format
                            - For fill-in-the-blank, use underscores appropriately

                            Return only the formatted Markdown.
                        """
        
        user_content = f"""Format this question:

Type: {question_type}
Question: {question_text}
Options: {options}

Current markdown: {question.get('markdown', '')}

Provide improved Markdown formatting."""
        
        messages = self._create_messages(system_prompt, user_content)
        improved_markdown = self._invoke_llm(messages)
        
        return improved_markdown.strip() if improved_markdown else question.get('markdown', '')
    
    def _analyze_question_metadata(self, question_text: str, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze question metadata"""
        
        metadata = {
            'required': False,
            'points': None,
            'difficulty': 'medium',
            'subject_area': 'general',
            'cognitive_level': 'comprehension'
        }
        
        # Check for required indicators
        if any(word in question_text.lower() for word in ['required', 'must', 'obligatorio']):
            metadata['required'] = True
        
        # Check for point values
        import re
        points_match = re.search(r'(\d+)\s*(?:point|pt|pts)', question_text.lower())
        if points_match:
            metadata['points'] = int(points_match.group(1))
        
        # Determine difficulty based on question complexity
        if len(options) > 5 or len(question_text) > 200:
            metadata['difficulty'] = 'hard'
        elif len(options) <= 2 or len(question_text) < 50:
            metadata['difficulty'] = 'easy'
        
        return metadata
    
    def _analyze_form_structure(self, form: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure and purpose of a complete form"""
        
        title = form.get('title', '')
        questions = form.get('questions', [])
        instructions = form.get('instructions', [])
        
        system_prompt = """Analyze this form and provide structural analysis.

Return analysis in JSON format:
{
    "form_type": "survey|exam|questionnaire|evaluation|registration|feedback",
    "purpose": "description of form purpose",
    "difficulty_level": "beginner|intermediate|advanced",
    "estimated_time": "time in minutes",
    "sections": ["section1", "section2"],
    "total_questions": number,
    "question_types_summary": {"type1": count, "type2": count}
}"""
        
        questions_summary = '\n'.join([
            f"{i+1}. {q.get('question', '')[:100]}..." 
            for i, q in enumerate(questions[:5])
        ])
        
        user_content = f"""Analyze this form:

Title: {title}
Instructions: {'; '.join(instructions)}
Total Questions: {len(questions)}

Sample Questions:
{questions_summary}

Provide structural analysis."""
        
        messages = self._create_messages(system_prompt, user_content)
        llm_response = self._invoke_llm(messages)
        
        # Parse analysis response
        analysis = self._parse_form_analysis(llm_response)
        
        return analysis
    
    def _generate_form_metadata(self, form: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for the form"""
        
        questions = form.get('questions', [])
        
        metadata = {
            'total_questions': len(questions),
            'question_types': {},
            'has_required_questions': False,
            'total_points': 0,
            'average_difficulty': 'medium'
        }
        
        # Count question types
        for question in questions:
            q_type = question.get('improved_type', question.get('type', 'unknown'))
            metadata['question_types'][q_type] = metadata['question_types'].get(q_type, 0) + 1
            
            # Check for required questions
            if question.get('required', False):
                metadata['has_required_questions'] = True
            
            # Sum points
            points = question.get('points')
            if points:
                metadata['total_points'] += points
        
        return metadata
    
    def _create_complete_form_markdown(self, form: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Create complete Markdown for the entire form"""
        
        lines = []
        
        # Form header
        title = form.get('title', 'Form')
        lines.append(f"# {title}")
        lines.append("")
        
        # Form metadata
        form_type = analysis.get('form_type', 'questionnaire')
        purpose = analysis.get('purpose', '')
        estimated_time = analysis.get('estimated_time', 'N/A')
        
        lines.append("## Form Information")
        lines.append(f"- **Type:** {form_type.title()}")
        if purpose:
            lines.append(f"- **Purpose:** {purpose}")
        lines.append(f"- **Estimated Time:** {estimated_time}")
        lines.append(f"- **Total Questions:** {len(form.get('questions', []))}")
        lines.append("")
        
        # Instructions
        instructions = form.get('instructions', [])
        if instructions:
            lines.append("## Instructions")
            for instruction in instructions:
                lines.append(f"- {instruction}")
            lines.append("")
        
        # Questions
        lines.append("## Questions")
        lines.append("")
        
        for i, question in enumerate(form.get('questions', []), 1):
            lines.append(f"### Question {i}")
            improved_markdown = question.get('improved_markdown', question.get('markdown', ''))
            lines.append(improved_markdown)
            lines.append("")
        
        return '\n'.join(lines)
    
    def _parse_form_analysis(self, response: str) -> Dict[str, Any]:
        """Parse form analysis response"""
        
        analysis = {
            'form_type': 'questionnaire',
            'purpose': '',
            'difficulty_level': 'intermediate',
            'estimated_time': '10-15 minutes',
            'sections': [],
            'total_questions': 0,
            'question_types_summary': {}
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
            self.logger.warning(f"Failed to parse form analysis response: {e}")
        
        return analysis
