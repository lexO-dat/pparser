"""
Image analysis and description agent
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from .base import BaseAgent
from ..extractors import ImageExtractor

from dotenv import load_dotenv
import os

# TODO: This agent is sort of implemented ( i recicled some code of other proyect ) but 4o-mini i think that does not do image analysis so is a little useless right now
# TODO: the temperature variable i think i will put it into the .env file to be more easy to change

TEMPERATURE = float(os.getenv('TEMPERATURE', 0.3))  # Default temperature if not set in .env

class ImageAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing images and generating descriptions"""
    
    def __init__(self, config, output_dir: Optional[Path] = None):
        super().__init__(
            config=config,
            name="ImageAnalysisAgent",
            role="Analyze images and generate descriptive content",
            temperature=TEMPERATURE
        )
        self.extractor = ImageExtractor(config=config, output_dir=output_dir)
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process PDF page and analyze images"""
        
        pdf_path = Path(input_data.get('pdf_path'))
        page_num = input_data.get('page_num', 0)
        
        # Extract images
        extraction_result = self.extractor.extract(pdf_path, page_num)
        
        if not extraction_result.get('images'):
            return {
                'success': True,
                'result': extraction_result,
                'agent': self.name,
                'message': 'No images found on this page'
            }
        
        # Enhance image descriptions with LLM analysis
        enhanced_result = self._enhance_image_analysis(extraction_result)
        
        return {
            'success': True,
            'result': enhanced_result,
            'agent': self.name
        }
    
    def _enhance_image_analysis(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance image analysis using LLM"""
        
        enhanced_images = []
        
        for image_info in extraction_result.get('images', []):
            enhanced_image = image_info.copy()
            
            # Generate enhanced description
            enhanced_description = self._generate_enhanced_description(image_info)
            enhanced_image['enhanced_description'] = enhanced_description
            
            # Determine image type and purpose
            image_classification = self._classify_image(image_info, enhanced_description)
            enhanced_image.update(image_classification)
            
            # Generate markdown reference
            enhanced_image['markdown_ref'] = self._generate_markdown_reference(enhanced_image)
            
            enhanced_images.append(enhanced_image)
        
        result = extraction_result.copy()
        result['images'] = enhanced_images
        
        return result
    
    def _generate_enhanced_description(self, image_info: Dict[str, Any]) -> str:
        """Generate enhanced description using LLM"""
        
        basic_description = image_info.get('description', '')
        filename = image_info.get('filename', '')
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        page_num = image_info.get('page_number', 1)
        
        system_prompt = """
                            You are an expert in document analysis and image description. Generate a comprehensive, descriptive alt-text for an image extracted from a PDF document.

                            Your description should:
                            1. Be concise but informative (1-2 sentences)
                            2. Include the likely purpose of the image in the document context
                            3. Mention any text, diagrams, charts, or key visual elements
                            4. Use descriptive language suitable for accessibility
                            5. Consider the academic/professional context of the document

                            Return only the description text, nothing else.
                        """
        
        user_content = f"""
                            Generate an enhanced description for this image:

                            Basic info: {basic_description}
                            Filename: {filename}
                            Dimensions: {width}x{height}
                            Page: {page_num}
                            Context: Image extracted from a PDF document

                            Provide a descriptive alt-text.
                        """
        
        messages = self._create_messages(system_prompt, user_content)
        enhanced_desc = self._invoke_llm(messages)
        
        return enhanced_desc.strip() if enhanced_desc else basic_description
    
    def _classify_image(self, image_info: Dict[str, Any], description: str) -> Dict[str, str]:
        """Classify image type and purpose"""
        
        system_prompt = """
                            Classify this image based on its description and context. Determine:

                            1. Image type: diagram, chart, graph, photo, illustration, screenshot, formula, table, figure, other
                            2. Purpose: decoration, explanation, data_visualization, reference, example, proof, other
                            3. Content category: mathematical, scientific, technical, business, educational, general

                            Return your classification in this format:
                            Type: [type]
                            Purpose: [purpose] 
                            Category: [category]
                        """
        
        user_content = f"""
                            Classify this image:
                            Description: {description}
                            Filename: {image_info.get('filename', '')}
                            Size: {image_info.get('width', 0)}x{image_info.get('height', 0)}
                        """
        
        messages = self._create_messages(system_prompt, user_content)
        classification_response = self._invoke_llm(messages)
        
        # Parse classification response
        classification = self._parse_classification(classification_response)
        
        return classification
    
    def _parse_classification(self, response: str) -> Dict[str, str]:
        """Parse LLM classification response"""
        
        classification = {
            'image_type': 'other',
            'purpose': 'other', 
            'category': 'general'
        }
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('Type:'):
                    classification['image_type'] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('Purpose:'):
                    classification['purpose'] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('Category:'):
                    classification['category'] = line.split(':', 1)[1].strip().lower()
        except Exception as e:
            self.logger.warning(f"Failed to parse image classification: {e}")
        
        return classification
    
    def _generate_markdown_reference(self, image_info: Dict[str, Any]) -> str:
        """Generate markdown reference for the image"""
        
        filename = image_info.get('filename', 'image.png')
        description = image_info.get('enhanced_description', image_info.get('description', 'Image'))
        
        # Clean description for alt text
        alt_text = description.replace('"', "'").replace('\n', ' ')
        
        return f'![{alt_text}](images/{filename})'


class ImagePositionAgent(BaseAgent):
    """Agent specialized in determining image placement in document structure"""
    
    def __init__(self):
        super().__init__(
            name="ImagePositionAgent",
            role="Determine optimal image placement in Markdown",
            temperature=0.1
        )
    
    # TODO: debug this, somethimes it does nothing :D
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Determine where images should be placed in the document"""
        
        images = input_data.get('images', [])
        text_structure = input_data.get('text_structure', {})
        page_num = input_data.get('page_num', 0)
        
        if not images:
            return {
                'success': True,
                'result': {'image_placements': []},
                'agent': self.name
            }
        
        # Analyze placement for each image
        placements = []
        for image in images:
            placement = self._determine_image_placement(image, text_structure)
            placements.append(placement)
        
        return {
            'success': True,
            'result': {'image_placements': placements},
            'agent': self.name
        }
    
    def _determine_image_placement(self, image: Dict[str, Any], text_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal placement for an image"""
        
        image_type = image.get('image_type', 'other')
        purpose = image.get('purpose', 'other')
        page_num = image.get('page_number', 1)
        
        # Default placement strategy
        placement = {
            'image': image,
            'placement_type': 'inline',  # inline, block, float
            'position': 'after_paragraph',  # before_section, after_paragraph, end_of_section
            'reference_needed': True,
            'caption': self._generate_caption(image)
        }
        
        # Adjust based on image type
        if image_type in ['chart', 'graph', 'diagram']:
            placement['placement_type'] = 'block'
            placement['position'] = 'after_paragraph'
        elif image_type in ['formula', 'equation']:
            placement['placement_type'] = 'inline'
            placement['position'] = 'inline'
        elif image_type == 'table':
            placement['placement_type'] = 'block'
            placement['position'] = 'after_paragraph'
        
        return placement
    
    def _generate_caption(self, image: Dict[str, Any]) -> str:
        """Generate a caption for the image"""
        
        description = image.get('enhanced_description', image.get('description', ''))
        image_type = image.get('image_type', 'Figure')
        page_num = image.get('page_number', 1)
        image_index = image.get('image_index', 0)
        
        # Generate figure number
        figure_num = f"{page_num}.{image_index + 1}"
        
        # Capitalize image type
        type_label = image_type.replace('_', ' ').title()
        if type_label.lower() == 'other':
            type_label = 'Figure'
        
        return f"{type_label} {figure_num}: {description}"
