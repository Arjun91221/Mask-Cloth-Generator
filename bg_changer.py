import os
import uuid
from PIL import Image#, ImageFile
from rembg.bg import remove
import numpy as np
from matplotlib import pyplot as plt
import cv2

current_directory = os.path.dirname(__file__)


def change_bg(org_img, bg_img):
    
    org_img = Image.open(org_img)
    bg_img = Image.open(bg_img)

    mask = remove(org_img, only_mask=True)
    plt.imshow(np.asarray(mask).astype(np.uint8))

    bg_img = bg_img.resize(org_img.size, Image.LANCZOS)
    result = Image.composite(org_img, bg_img, mask)
    result.convert("RGB")

    output_path = f'{current_directory}/uploads/{str(uuid.uuid4())}.png'
    result.save(output_path)

    return output_path

def perfect_change_bg(org_img, bg_img):

    image = cv2.imread(org_img)
    image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lower_green = np.array([0, 150, 0], dtype=np.uint8)
    upper_green = np.array([125, 255, 120], dtype=np.uint8)

    mask = cv2.inRange(image_copy, lower_green, upper_green)

    background_change = np.copy(image_copy)
    background_change[mask != 0] = [255, 255, 255]

    masked_image = np.copy(image_copy)
    masked_image[mask != 0] = [0, 0, 0]

    background_image = cv2.imread(bg_img)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

    crop_background = cv2.resize(background_image, (masked_image.shape[1], masked_image.shape[0]))
    final_image = cv2.add(masked_image,crop_background)

    output_path = f'{current_directory}/uploads/{str(uuid.uuid4())}.png'

    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)  
    cv2.imwrite(output_path, final_image_bgr)

    return output_path
    