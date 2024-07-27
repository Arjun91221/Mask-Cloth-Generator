import os
from typing import Tuple, List
import uuid
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

current_directory = os.path.dirname(__file__)

class ClothSegmenter:
    """
    A class for segmenting and extracting clothes from images.
    """

    def __init__(self, model_name: str = "mattmdjaga/segformer_b2_clothes"):
        """
        Initialize the ClothSegmenter with a pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model to use.
        """
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.clothes_labels = [4, 5, 6, 7, 8, 17]  # Upper-clothes, Skirt, Pants, Dress, Belt, Scarf
        self.clothes_labels.extend([9, 10]) # Left shoe, Right shoe # Comment this like to not mask footwear

    def generate_cloth_mask(self, image_path: str) -> np.ndarray:
        """
        Generate a binary mask for clothing regions in the input image.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            np.ndarray: Binary mask of clothing regions.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=True,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        clothes_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
        for label in self.clothes_labels:
            clothes_mask |= (pred_seg == label)

        return clothes_mask.numpy().astype("uint8") * 255

    def extract_clothes(self, image_path: str, background_color: Tuple[int, int, int] = (240, 240, 240)) -> Image.Image:
        """
        Extract clothes from the input image and place them on a custom background.

        Args:
            image_path (str): Path to the input image file.
            background_color (Tuple[int, int, int]): RGB color for the background.

        Returns:
            Image.Image: Image of extracted clothes on the specified background.
        """
        original_image = Image.open(image_path).convert("RGBA")
        cloth_mask = self.generate_cloth_mask(image_path)

        mask_rgba = Image.fromarray(cloth_mask).convert("L")

        background = Image.new("RGBA", original_image.size, background_color + (255,))

        extracted_clothes = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        extracted_clothes.paste(original_image, (0, 0), mask_rgba)

        final_image = Image.alpha_composite(background, extracted_clothes)

        return final_image
    
def main(image_path):
    segmenter = ClothSegmenter()
    extracted_cloth = segmenter.extract_clothes(image_path)
    output_path = f'{current_directory}/uploads/{str(uuid.uuid4())}.png'
    extracted_cloth.save(output_path , format='PNG')
    return output_path

