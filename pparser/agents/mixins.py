"""
Mixins and utilities for agent functionality to reduce code duplication.
"""

from typing import Dict, Any, List, Optional
import json
import re
from abc import ABC

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from ..utils.logger import get_logger


class LLMInteractionMixin:
    """Mixin for standardized LLM interactions across agents."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def create_standard_messages(self, system_prompt: str, user_content: str, 
                                context: Optional[Dict] = None) -> List[BaseMessage]:
        """Create standardized message structure for LLM interactions."""
        messages = [SystemMessage(content=system_prompt)]
        
        if context:
            context_str = f"Context: {json.dumps(context, indent=2)}\n\n"
            user_content = context_str + user_content
            
        messages.append(HumanMessage(content=user_content))
        return messages
    
    def invoke_llm_with_retry(self, messages: List[BaseMessage], 
                             parse_json: bool = False, 
                             max_retries: int = 3) -> Optional[str]:
        """Invoke LLM with retry logic and error handling."""
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                content = response.content.strip()
                
                if parse_json:
                    try:
                        json.loads(content)
                        return content
                    except json.JSONDecodeError:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Invalid JSON on attempt {attempt + 1}, retrying...")
                            continue
                        else:
                            self.logger.error("Failed to get valid JSON after retries")
                            return None
                
                return content
                
            except Exception as e:
                self.logger.error(f"LLM invocation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return None
        
        return None
    
    def extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response with fallback parsing."""
        try:
            # Try direct JSON parsing
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # Try to find JSON object in text
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(response[start:end])
                    
            except json.JSONDecodeError:
                pass
        
        self.logger.warning("Could not extract JSON from response")
        return None


class ContentFormattingMixin:
    """Mixin for common content formatting operations."""
    
    @staticmethod
    def clean_markdown_code_blocks(text: str) -> str:
        """Remove markdown code block markers from text."""
        if text.startswith('```markdown'):
            text = text[11:-3] if text.endswith('```') else text[11:]
        elif text.startswith('```'):
            text = text[3:-3] if text.endswith('```') else text[3:]
        return text.strip()
    
    @staticmethod
    def fix_math_delimiters(text: str) -> str:
        """Fix LaTeX math delimiters to use Markdown format."""
        # Convert \( ... \) to $ ... $
        text = re.sub(r'\\\(([^)]*)\\\)', r'$\1$', text)
        
        # Convert \[ ... \] to $$ ... $$
        text = re.sub(r'\\\[([^]]*)\\\]', r'$$\1$$', text)
        
        return text
    
    @staticmethod
    def truncate_for_prompt(text: str, max_length: int = 1000) -> str:
        """Truncate text for use in prompts with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    @staticmethod
    def create_section_anchor(title: str) -> str:
        """Create URL-safe anchor from section title."""
        return title.lower().replace(' ', '-').replace('&', 'and')


class ValidationMixin:
    """Mixin for content validation operations."""
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate that all required fields are present in data."""
        return all(field in data and data[field] for field in required_fields)
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 10000) -> str:
        """Sanitize and limit input text length."""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text


class PromptTemplateFactory:
    """Factory for creating standardized prompts across agents."""
    
    @staticmethod
    def create_analysis_prompt(content_type: str, 
                              analysis_goals: List[str],
                              output_format: str = "JSON") -> str:
        """Create standardized analysis prompt template."""
        goals_text = '\n'.join(f"{i+1}. {goal}" for i, goal in enumerate(analysis_goals))
        
        return f"""You are an expert in {content_type} analysis. Your task is to analyze the provided content and provide detailed insights.

Analysis Goals:
{goals_text}

Requirements:
- Provide comprehensive analysis
- Be specific and actionable in recommendations
- Maintain professional tone
- Return results in {output_format} format
- Focus on document structure and readability

Return only the analysis in the requested format, no additional commentary."""
    
    @staticmethod
    def create_formatting_prompt(content_type: str, 
                                formatting_rules: List[str]) -> str:
        """Create standardized formatting prompt template."""
        rules_text = '\n'.join(f"- {rule}" for rule in formatting_rules)
        
        return f"""You are an expert in {content_type} formatting. Your task is to improve the formatting and structure of the provided content.

Formatting Rules:
{rules_text}

Requirements:
- Maintain original meaning and intent
- Improve readability and structure
- Use proper Markdown syntax
- Ensure consistent formatting
- Return only the formatted content, no explanations"""
    
    @staticmethod
    def create_classification_prompt(item_type: str, 
                                   classification_categories: Dict[str, List[str]]) -> str:
        """Create standardized classification prompt template."""
        categories_text = ""
        for category, options in classification_categories.items():
            options_text = ', '.join(options)
            categories_text += f"- {category}: {options_text}\n"
        
        return f"""Classify this {item_type} according to the following categories:

{categories_text}

Return your classification in JSON format with each category as a key."""
