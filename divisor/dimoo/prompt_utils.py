"""
Prompt generation utilities for different inference types
"""

from typing import Dict, List, Tuple, Optional


def create_prompt_templates() -> dict:
    """Create prompt templates for various tasks"""
    templates = {
        "text_to_text": "Generate text according to the text prompt",
        "text_understanding": "You are a multimodal model that can process both text and images. Answer the following question based on the provided images. Analyze each image and combine relevant details to answer.",
        "image_generation": "Generate an image according to the text prompt.",
        "image_editing": "Generate an image applying the following editing instruction based on the original image.",
        "dense_prediction": "Perform dense prediction on the given images.",
        "control_generation": "Generate an image according to the text prompt and the given control image.",
        "subject_generation": "Generate an image according to the text prompt and the given object image.",
        "multi_view": "Generate a view-image based on the given image.",
        "style_transfer": "Transform the current image into the style of the provided image.",
    }
    return templates


def generate_multimodal_understanding_prompt(question: str, templates: Optional[Dict] = None) -> str:
    """Generate prompt for multimodal understanding (MMU)\n
    :param question: User question about the image
    :param templates: Optional prompt templates dict
    :return: Formatted input prompt"""

    if templates is None:
        templates = create_prompt_templates()

    system_prompt = templates["text_understanding"]
    input_prompt = "<system>" + system_prompt + "</system>" + "<user>" + question + "</user>"

    return input_prompt


def generate_text_prompt(question: str, templates: dict = create_prompt_templates()) -> str:
    """Generate prompt for text to text generation\n
    :param question: User question
    :param templates: Optional prompt templates dict
    :return: Formatted input prompt"""

    system_prompt = templates["text_to_text"]
    input_prompt = "<system>" + system_prompt + "</system>" + "<user>" + question + "</user>"

    return input_prompt
