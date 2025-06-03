"""
Quality validation agent for verifying and improving output quality.

This module contains the QualityValidator agent that performs comprehensive
quality checks on the generated Markdown and suggests improvements.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .base import BaseAgent
from ..config import Config
from ..utils.logger import get_logger

from dotenv import load_dotenv
import os

logger = get_logger(__name__)

load_dotenv()
TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', 0.1))

class QualityValidatorAgent(BaseAgent):
    """
    Agent responsible for validating and improving the quality of generated
    Markdown documents and ensuring faithful reproduction of PDF content.
    """

    def __init__(self, config):
        """
        Initialize the quality validator agent.
        
        Args:
            config: Configuration object containing settings
        """
        super().__init__(
            config=config,
            name="QualityValidatorAgent", 
            role="Validate and improve the quality of generated Markdown"
        )
        self.llm = ChatOpenAI(
            model=self.config.openai_model,
            temperature=0.1,
            max_tokens=4000,
            openai_api_key=self.config.openai_api_key
        )

    async def process(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and improve the quality of generated Markdown.
        
        Args:
            validation_data: Dictionary containing markdown, original content, and metadata
            
        Returns:
            Dictionary containing validation results and improved content
        """
        try:
            self.logger.info("Starting quality validation")
            
            markdown = validation_data.get('markdown', '')
            original_content = validation_data.get('original_content', {})
            structure_map = validation_data.get('structure_map', {})
            
            # Perform various quality checks
            checks = await self._perform_quality_checks(markdown, original_content, structure_map)
            
            # Generate improvement suggestions
            improvements = await self._generate_improvements(markdown, checks)
            
            # Apply automatic fixes if enabled
            improved_markdown = await self._apply_automatic_fixes(markdown, improvements)
            
            # Final validation score
            score = self._calculate_quality_score(checks)
            
            return {
                'status': 'success',
                'quality_score': score,
                'validation_checks': checks,
                'improvements': improvements,
                'original_markdown': markdown,
                'improved_markdown': improved_markdown,
                'recommendations': await self._generate_recommendations(checks, score)
            }
            
        except Exception as e:
            self.logger.error(f"Error in quality validation: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'quality_score': 0,
                'validation_checks': {},
                'improvements': [],
                'original_markdown': validation_data.get('markdown', ''),
                'improved_markdown': validation_data.get('markdown', '')
            }

    async def _perform_quality_checks(
        self, 
        markdown: str, 
        original_content: Dict[str, Any], 
        structure_map: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive quality checks on the Markdown."""
        
        checks = {
            'structure_quality': await self._check_structure_quality(markdown),
            'content_completeness': await self._check_content_completeness(markdown, original_content),
            'formatting_quality': self._check_formatting_quality(markdown),
            'asset_integrity': self._check_asset_integrity(markdown, structure_map),
            'readability': await self._check_readability(markdown),
            'accuracy': await self._check_accuracy(markdown, original_content)
        }
        
        return checks

    async def _check_structure_quality(self, markdown: str) -> Dict[str, Any]:
        """Check the structural quality of the Markdown."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a document structure analyst. Evaluate the structural quality of this Markdown document.

Check for:
1. Proper heading hierarchy (no skipped levels)
2. Logical section organization
3. Appropriate use of Markdown elements
4. Consistent formatting
5. Good document flow

Return JSON with:
- score: 0-100
- issues: Array of structural issues found
- strengths: Array of structural strengths
- suggestions: Array of improvement suggestions"""),
            
            HumanMessage(content=f"Analyze this Markdown document structure:\n\n{markdown[:3000]}...")
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            result_text = response.content.strip()
            
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
            
            return json.loads(result_text)
            
        except Exception as e:
            self.logger.error(f"Error checking structure quality: {str(e)}")
            return {
                'score': 50,
                'issues': ['Structure analysis failed'],
                'strengths': [],
                'suggestions': []
            }

    async def _check_content_completeness(self, markdown: str, original_content: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all original content is represented in the Markdown."""
        
        completeness = {
            'score': 0,
            'missing_content': [],
            'preserved_content': [],
            'coverage_by_type': {}
        }
        
        try:
            # Check text content preservation
            if 'text' in original_content:
                text_coverage = self._analyze_text_coverage(markdown, original_content['text'])
                completeness['coverage_by_type']['text'] = text_coverage
            
            # Check asset references
            for asset_type in ['images', 'tables', 'formulas', 'forms']:
                if asset_type in original_content:
                    asset_coverage = self._analyze_asset_coverage(markdown, original_content[asset_type], asset_type)
                    completeness['coverage_by_type'][asset_type] = asset_coverage
            
            # Calculate overall score
            type_scores = [coverage.get('score', 0) for coverage in completeness['coverage_by_type'].values()]
            completeness['score'] = sum(type_scores) / len(type_scores) if type_scores else 0
            
        except Exception as e:
            self.logger.error(f"Error checking content completeness: {str(e)}")
            completeness['score'] = 50
        
        return completeness

    def _check_formatting_quality(self, markdown: str) -> Dict[str, Any]:
        """Check the formatting quality of the Markdown."""
        
        issues = []
        strengths = []
        score = 100
        
        # Check for common formatting issues
        lines = markdown.split('\n')
        
        # Check heading consistency
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        heading_levels = []
        
        for line in lines:
            match = heading_pattern.match(line)
            if match:
                level = len(match.group(1))
                heading_levels.append(level)
        
        # Check for skipped heading levels
        if heading_levels:
            for i in range(1, len(heading_levels)):
                if heading_levels[i] > heading_levels[i-1] + 1:
                    issues.append(f"Skipped heading level: from H{heading_levels[i-1]} to H{heading_levels[i]}")
                    score -= 10
        
        # Check for proper spacing
        empty_line_before_heading = True
        for i, line in enumerate(lines):
            if heading_pattern.match(line):
                if i > 0 and lines[i-1].strip() != '':
                    empty_line_before_heading = False
                    break
        
        if not empty_line_before_heading:
            issues.append("Missing empty lines before headings")
            score -= 5
        else:
            strengths.append("Proper spacing before headings")
        
        # Check for consistent list formatting
        list_patterns = [r'^\s*[-*+]\s+', r'^\s*\d+\.\s+']
        for pattern in list_patterns:
            if re.search(pattern, markdown, re.MULTILINE):
                strengths.append("Uses proper list formatting")
                break
        
        # Check for proper table formatting
        if '|' in markdown:
            table_lines = [line for line in lines if '|' in line]
            if len(table_lines) >= 2:
                strengths.append("Contains properly formatted tables")
            else:
                issues.append("Potential table formatting issues")
                score -= 5
        
        return {
            'score': max(0, score),
            'issues': issues,
            'strengths': strengths,
            'suggestions': [f"Fix: {issue}" for issue in issues]
        }

    def _check_asset_integrity(self, markdown: str, structure_map: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all assets are properly referenced and formatted."""
        
        integrity = {
            'score': 100,
            'missing_assets': [],
            'broken_references': [],
            'properly_referenced': []
        }
        
        try:
            assets = structure_map.get('assets', {})
            
            for asset_type, asset_list in assets.items():
                for asset in asset_list:
                    asset_id = asset.get('id', '')
                    
                    # Check if asset is referenced in markdown
                    if asset_type == 'images':
                        # Look for image references
                        if re.search(rf'!\[.*?\]\(.*?{re.escape(asset_id)}.*?\)', markdown):
                            integrity['properly_referenced'].append(f"{asset_type}: {asset_id}")
                        else:
                            integrity['missing_assets'].append(f"{asset_type}: {asset_id}")
                            integrity['score'] -= 10
                    
                    elif asset_type == 'tables':
                        # Look for table content or references
                        if asset_id in markdown or 'table' in markdown.lower():
                            integrity['properly_referenced'].append(f"{asset_type}: {asset_id}")
                        else:
                            integrity['missing_assets'].append(f"{asset_type}: {asset_id}")
                            integrity['score'] -= 10
                    
                    elif asset_type == 'formulas':
                        # Look for formula references or LaTeX
                        if '$' in markdown or 'formula' in markdown.lower():
                            integrity['properly_referenced'].append(f"{asset_type}: {asset_id}")
                        else:
                            integrity['missing_assets'].append(f"{asset_type}: {asset_id}")
                            integrity['score'] -= 10
                    
                    elif asset_type == 'forms':
                        # Look for form elements
                        if re.search(r'\[.*?\]\(.*?\)', markdown) or 'form' in markdown.lower():
                            integrity['properly_referenced'].append(f"{asset_type}: {asset_id}")
                        else:
                            integrity['missing_assets'].append(f"{asset_type}: {asset_id}")
                            integrity['score'] -= 10
        
        except Exception as e:
            self.logger.error(f"Error checking asset integrity: {str(e)}")
            integrity['score'] = 50
        
        integrity['score'] = max(0, integrity['score'])
        return integrity

    async def _check_readability(self, markdown: str) -> Dict[str, Any]:
        """Check the readability of the Markdown document."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a readability expert. Evaluate the readability of this Markdown document.

Consider:
1. Sentence length and complexity
2. Paragraph structure
3. Use of headings and sections
4. Flow and transitions
5. Overall clarity

Return JSON with:
- score: 0-100
- readability_level: "easy", "moderate", "difficult"
- issues: Array of readability issues
- suggestions: Array of improvement suggestions"""),
            
            HumanMessage(content=f"Analyze the readability of this document:\n\n{markdown[:2000]}...")
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            result_text = response.content.strip()
            
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
            
            return json.loads(result_text)
            
        except Exception as e:
            self.logger.error(f"Error checking readability: {str(e)}")
            return {
                'score': 75,
                'readability_level': 'moderate',
                'issues': ['Readability analysis failed'],
                'suggestions': []
            }

    async def _check_accuracy(self, markdown: str, original_content: Dict[str, Any]) -> Dict[str, Any]:
        """Check the accuracy of content representation."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an accuracy validator. Compare the Markdown output with the original content data to check for accuracy.

                                    Evaluate:
                                    1. Factual accuracy of converted content
                                    2. Preservation of key information
                                    3. Correct representation of data structures
                                    4. Maintenance of original meaning

                                    Return JSON with:
                                    - score: 0-100
                                    - accuracy_issues: Array of accuracy problems found
                                    - preserved_elements: Array of well-preserved content elements
                                    - recommendations: Array of accuracy improvement suggestions"""),
            
            HumanMessage(content=f"""
                                    Compare this Markdown output with the original content:

                                    MARKDOWN OUTPUT:
                                    {markdown[:2000]}...

                                    ORIGINAL CONTENT SUMMARY:
                                    {json.dumps({k: f"{type(v).__name__} with {len(v) if isinstance(v, (list, dict)) else 'content'}" for k, v in original_content.items()}, indent=2)}

                                    Evaluate accuracy and preservation of original content.
                                """)])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            result_text = response.content.strip()
            
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
            
            return json.loads(result_text)
            
        except Exception as e:
            self.logger.error(f"Error checking accuracy: {str(e)}")
            return {
                'score': 75,
                'accuracy_issues': ['Accuracy analysis failed'],
                'preserved_elements': [],
                'recommendations': []
            }

    async def _generate_improvements(self, markdown: str, checks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific improvement suggestions based on quality checks."""
        
        improvements = []
        
        # Structure improvements
        structure_check = checks.get('structure_quality', {})
        for suggestion in structure_check.get('suggestions', []):
            improvements.append({
                'type': 'structure',
                'description': suggestion,
                'priority': 'medium',
                'auto_fixable': False
            })
        
        # Formatting improvements
        formatting_check = checks.get('formatting_quality', {})
        for suggestion in formatting_check.get('suggestions', []):
            improvements.append({
                'type': 'formatting',
                'description': suggestion,
                'priority': 'low',
                'auto_fixable': True
            })
        
        # Asset improvements
        asset_check = checks.get('asset_integrity', {})
        for missing_asset in asset_check.get('missing_assets', []):
            improvements.append({
                'type': 'asset',
                'description': f"Add missing asset reference: {missing_asset}",
                'priority': 'high',
                'auto_fixable': False
            })
        
        # Readability improvements
        readability_check = checks.get('readability', {})
        for suggestion in readability_check.get('suggestions', []):
            improvements.append({
                'type': 'readability',
                'description': suggestion,
                'priority': 'medium',
                'auto_fixable': False
            })
        
        # Accuracy improvements
        accuracy_check = checks.get('accuracy', {})
        for recommendation in accuracy_check.get('recommendations', []):
            improvements.append({
                'type': 'accuracy',
                'description': recommendation,
                'priority': 'high',
                'auto_fixable': False
            })
        
        return improvements

    async def _apply_automatic_fixes(self, markdown: str, improvements: List[Dict[str, Any]]) -> str:
        """Apply automatic fixes for simple formatting issues."""
        
        fixed_markdown = markdown
        
        for improvement in improvements:
            if improvement.get('auto_fixable', False) and improvement['type'] == 'formatting':
                description = improvement['description']
                
                # Fix spacing before headings
                if "empty lines before headings" in description.lower():
                    lines = fixed_markdown.split('\n')
                    new_lines = []
                    
                    for i, line in enumerate(lines):
                        if re.match(r'^#{1,6}\s+', line) and i > 0 and lines[i-1].strip() != '':
                            new_lines.append('')  # Add empty line
                        new_lines.append(line)
                    
                    fixed_markdown = '\n'.join(new_lines)
        
        return fixed_markdown

    def _calculate_quality_score(self, checks: Dict[str, Any]) -> float:
        """Calculate overall quality score from all checks."""
        
        scores = []
        weights = {
            'structure_quality': 0.25, # TODO: see why this is failing
            'content_completeness': 0.30,
            'formatting_quality': 0.15,
            'asset_integrity': 0.15,
            'readability': 0.10,
            'accuracy': 0.25
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for check_name, weight in weights.items():
            if check_name in checks:
                score = checks[check_name].get('score', 0)
                weighted_sum += score * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0

    async def _generate_recommendations(self, checks: Dict[str, Any], score: float) -> List[str]:
        """Generate high-level recommendations based on validation results."""
        
        recommendations = []
        
        if score < 60:
            recommendations.append("Document quality is below acceptable threshold. Major revisions needed.")
        elif score < 80:
            recommendations.append("Document quality is good but could benefit from improvements.")
        else:
            recommendations.append("Document quality is excellent.")
        
        # Specific recommendations based on lowest scoring areas
        lowest_score = 100
        lowest_area = None
        
        for check_name, check_data in checks.items():
            check_score = check_data.get('score', 0)
            if check_score < lowest_score:
                lowest_score = check_score
                lowest_area = check_name
        
        if lowest_area and lowest_score < 70:
            area_names = {
                'structure_quality': 'document structure',
                'content_completeness': 'content completeness',
                'formatting_quality': 'formatting',
                'asset_integrity': 'asset references',
                'readability': 'readability',
                'accuracy': 'content accuracy'
            }
            recommendations.append(f"Focus improvement efforts on {area_names.get(lowest_area, lowest_area)}.")
        
        return recommendations

    def _analyze_text_coverage(self, markdown: str, original_text: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well the original text content is covered in the Markdown."""
        
        # Simple word-based coverage analysis
        original_words = set()
        if 'content' in original_text:
            content = str(original_text['content']).lower()
            original_words = set(re.findall(r'\b\w+\b', content))
        
        markdown_words = set(re.findall(r'\b\w+\b', markdown.lower()))
        
        if original_words:
            coverage = len(original_words.intersection(markdown_words)) / len(original_words)
        else:
            coverage = 1.0
        
        return {
            'score': coverage * 100,
            'original_words': len(original_words),
            'preserved_words': len(original_words.intersection(markdown_words)),
            'coverage_ratio': coverage
        }

    def _analyze_asset_coverage(self, markdown: str, assets: Dict[str, Any], asset_type: str) -> Dict[str, Any]:
        """Analyze coverage of specific asset type in the Markdown."""
        
        if not isinstance(assets, dict) or 'items' not in assets:
            return {'score': 100, 'total_assets': 0, 'referenced_assets': 0}
        
        total_assets = len(assets['items'])
        referenced_assets = 0
        
        for asset in assets['items']:
            asset_id = asset.get('id', '')
            if asset_id and asset_id in markdown:
                referenced_assets += 1
        
        coverage = referenced_assets / total_assets if total_assets > 0 else 1.0
        
        return {
            'score': coverage * 100,
            'total_assets': total_assets,
            'referenced_assets': referenced_assets,
            'coverage_ratio': coverage
        }
