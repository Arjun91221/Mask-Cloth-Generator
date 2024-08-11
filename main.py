import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse
import uuid
import aiohttp
from io import BytesIO
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import io
from cloth_segementation import main
from bg_changer import change_bg, perfect_change_bg
from typing import List

app = FastAPI()

current_directory = os.path.dirname(__file__)

UPLOAD_DIR = f"{current_directory}/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class Base64Image(BaseModel):
    base64_string: str

async def fetch_image_from_url(url: str) -> io.BytesIO:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Invalid URL")
            return BytesIO(await response.read())
        
def decode_base64_image(base64_string: str) -> BytesIO:
    image_data = base64.b64decode(base64_string)
    return BytesIO(image_data)
        
@app.post("/image-to-base64/")
async def image_to_base64(file: UploadFile = File(None), url: str = Form(None)):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="No image file or URL provided")

    if file:
        contents = await file.read()
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(contents)
    else:
        image_io = await fetch_image_from_url(url)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    os.remove(image_path)
    
    return encoded_string

@app.post("/base64-to-image/")
async def base64_to_image(base64_image: Base64Image):
    try:
        base64_string = base64_image.base64_string
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        image.save(image_path)
        
        return StreamingResponse(open(image_path, "rb"), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")

def initialize_model():
    try:
        # Load YOLOv7 model
        model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model=f'{current_directory}/yolov7/yolov7.pt')
        print("YOLOv7 model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOv7 model: {e}")
        model = None

    return model

# Initialize models with retry mechanism
yolo_model = initialize_model()

# If the initialization fails, try again
if yolo_model is None:
    print("Retrying model initialization...")
    yolo_model = initialize_model()

if yolo_model is None:
    raise("Failed to initialize model after retrying.")
else:
    print("Model initialized successfully.")

# Load OpenCV's DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    f'{current_directory}/models/deploy.prototxt',
    f'{current_directory}/models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
)

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.15)


def to_preserve_mask(image_path):
    """
    Generate a binary mask for clothing regions in the input image.

    Args:
    image_path (str): Path to the input image file.

    Returns:
    numpy.ndarray: Binary mask of clothing regions.
    """

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=True,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    clothes_labels = [1, 2, 3, 11, 16]  # Hat, Hair, Sunglasses, left footwear, right footwear, face, bag
    preserve_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
    for label in clothes_labels:
        preserve_mask |= (pred_seg == label)
    preserve_mask_image = preserve_mask.cpu().numpy().astype("uint8") * 255

    return preserve_mask_image

def get_arm_mask(image_path):
    """
    Generate a binary mask for clothing regions in the input image.

    Args:
    image_path (str): Path to the input image file.

    Returns:
    numpy.ndarray: Binary mask of clothing regions.
    """

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=True,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    clothes_labels = [14, 15]  # left arm, right arm
    arm_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
    for label in clothes_labels:
        arm_mask |= (pred_seg == label)
    arm_mask_image = arm_mask.cpu().numpy().astype("uint8") * 255

    return arm_mask_image

def feet_masking(image_path, smooth=True):
    """
    Generate a binary mask for feet and lower leg regions in the input image,
    and smooth the edges to create an aesthetically pleasing mask.

    Args:
    image_path (str): Path to the input image file.
    smooth (bool): Whether to smooth the feet mask. Default is True.

    Returns:
    numpy.ndarray: Binary mask of feet and lower leg regions with smoothed edges.
    """

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=(image.shape[0], image.shape[1]),
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    footwear_leg_labels = [9, 10, 12, 13]  # Left-shoe, Right-shoe, Left-leg, Right-leg
    footwear_leg_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
    for label in footwear_leg_labels:
        footwear_leg_mask |= (pred_seg == label)
    footwear_leg_mask_image = footwear_leg_mask.cpu().numpy().astype("uint8") * 255

    results = pose.process(image_rgb)

    if results.pose_landmarks:
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        left_ankle_y = int(left_ankle.y * image.shape[0])
        right_ankle_y = int(right_ankle.y * image.shape[0])

        highest_ankle_y = min(left_ankle_y, right_ankle_y)
        footwear_leg_mask_image[:highest_ankle_y, :] = 0

    if smooth:
        # Smooth the mask edges
        blurred_mask = cv2.GaussianBlur(footwear_leg_mask_image, (15, 15), 0)

        # Find contours
        contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smooth_mask = np.zeros_like(footwear_leg_mask_image)

        # Fit ellipse to the contours and draw it on the smooth_mask
        for contour in contours:
            if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(smooth_mask, ellipse, (255), thickness=-1)

        return smooth_mask
    else:
        return footwear_leg_mask_image
    

def palm_masking(image_path, scale_factor=1.15, conf_threshold=0.5):
    """
    Generate a binary mask for palm regions in the input image with an option to scale the mask.
    If fewer than two hands are detected, use YOLOv7 to detect the human and crop the image to try hand detection again.
    Also, return the original image with landmarks drawn.

    Args:
    image_path (str): Path to the input image file.
    scale_factor (float): Factor to scale the size of the palm mask. Default is 1.15 (15% larger).
    conf_threshold (float): Confidence threshold for YOLOv7 human detection. Default is 0.5.

    Returns:
    tuple: Binary mask of palm regions, original image with landmarks drawn, or a message if hands are not visible.
    """

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_image = image_rgb.copy()
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    def detect_hands(image_rgb, x_offset=0, y_offset=0):
        results = hands.process(image_rgb)
        hand_landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w = image_rgb.shape[:2]
                points = [(int(landmark.x * w) + x_offset, int(landmark.y * h) + y_offset) for landmark in hand_landmarks.landmark]
                points = np.array(points, dtype=np.int32)
                if len(points) >= 5:
                    hand_landmarks_list.append(points)
                    # Draw landmarks on the annotated image
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        cv2.line(annotated_image,
                                 (points[start_idx][0], points[start_idx][1]),
                                 (points[end_idx][0], points[end_idx][1]),
                                 (0, 255, 0), 2)
                    for point in points:
                        cv2.circle(annotated_image, (point[0], point[1]), 5, (0, 0, 255), -1)
        return hand_landmarks_list

    def create_mask(hand_landmarks_list):
        for points in hand_landmarks_list:
            ellipse = cv2.fitEllipse(points)
            center, axes, angle = ellipse
            axes = (int(axes[0] * scale_factor), int(axes[1] * scale_factor))
            cv2.ellipse(mask, (center, axes, angle), 255, -1)

    # First attempt to detect hands
    print("First attempt to detect hands")
    hand_landmarks_list = detect_hands(image_rgb)

    if len(hand_landmarks_list) < 2:
        # Use YOLOv7 to detect human and crop image if fewer than 2 hands are detected
        print("Second attempt to detect hands")
        image_rgb_copy = image_rgb.copy()
        results = yolo_model(image_rgb)
        detections = results.xyxy[0].cpu().numpy()
        max_conf = 0
        best_box = None

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 0 and conf > conf_threshold and conf > max_conf:  # Check if it's a person and above confidence threshold
                max_conf = conf
                best_box = (int(x1), int(y1), int(x2), int(y2))

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cropped_image = image_rgb_copy[y1:y2, x1:x2]
            hand_landmarks_list = detect_hands(cropped_image, x_offset=x1, y_offset=y1)

    if not hand_landmarks_list:
        return "hands not visible", annotated_image

    create_mask(hand_landmarks_list)

    return mask, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


def detect_humans(image_path, conf_threshold=0.5):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Perform inference with YOLOv7
    results = yolo_model(image_rgb)
    detections = results.xyxy[0].cpu().numpy()

    max_conf = 0
    best_box = None

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) == 0 and conf > conf_threshold and conf > max_conf:  # Check if it's a person and above confidence threshold
            max_conf = conf
            best_box = (int(x1), int(y1), int(x2), int(y2))

    face_found = False
    if best_box is not None:
        x1, y1, x2, y2 = best_box

        # Perform face detection with OpenCV DNN
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                face_x1, face_y1, face_x2, face_y2 = box.astype("int")

                # Draw the face bounding box for debugging
                img_copy = image_rgb.copy()
                cv2.rectangle(img_copy, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)
                face_found = True

                # Adjust the y1 coordinate of the bounding box to exclude the neck and above
                y1 = int(face_y2-0.1*(face_y2-face_y1))

                # Display the image with face detection for debugging
                plt.figure(figsize=(10, 10))
                plt.imshow(img_copy)
                plt.title("Face Detection")
                plt.axis('off')
                plt.show()

        if not face_found:
            # Use Mediapipe pose to get landmarks if no face is found
            print("Face not found, going with mediapipe.")
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * h
                left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h

                avg_y = int((3*nose_y + left_shoulder_y + right_shoulder_y) / 5)
                y1 = avg_y

        # Draw the adjusted bounding box
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the image with Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    return image_rgb

def generate_body_mask(image_path, conf_threshold=0.5, display=False):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    mask = np.zeros(shape=(h,w), dtype=np.uint8)

    # Perform inference with YOLOv7
    results = yolo_model(image_rgb)
    detections = results.xyxy[0].cpu().numpy()

    max_conf = 0
    best_box = None

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) == 0 and conf > conf_threshold and conf > max_conf:  # Check if it's a person and above confidence threshold
            max_conf = conf
            best_box = (int(x1), int(y1), int(x2), int(y2))

    face_found = False

    if best_box is not None:
        x1, y1, x2, y2 = best_box

        # Perform face detection with OpenCV DNN
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                face_x1, face_y1, face_x2, face_y2 = box.astype("int")

                # Adjust the y2 coordinate of the bounding box to exclude the neck and above
                y1 = int(face_y2-0.1*(face_y2-face_y1))

                face_found = True

        if not face_found:
            # Use Mediapipe pose to get landmarks if no face is found
            print("Face not found, going with mediapipe.")
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y * h
                left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h

                avg_y = int((3*nose_y + left_shoulder_y + right_shoulder_y) / 5)
                y1 = avg_y

        # Draw the adjusted bounding box
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        mask[y1:y2, x1:x2] = 255
    if display:
      # Display the image with Matplotlib
      plt.figure(figsize=(10, 10))
      plt.imshow(image_rgb)
      plt.axis('off')
      plt.show()

    return mask

def generate_cloth_mask(image_path: str) -> np.ndarray:
    """
    Generate a binary mask for clothing regions in the input image.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        np.ndarray: Binary mask of clothing regions.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=True,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    clothes_labels = [4, 5, 6, 7, 8, 17]  # Upper-clothes, Skirt, Pants, Dress, Belt, Scarf

    clothes_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
    for label in clothes_labels:
        clothes_mask |= (pred_seg == label)

    return clothes_mask.cpu().numpy().astype("uint8") * 255

def custom_bitwise_op(A, B):
  if isinstance(A, str) and isinstance(B, str):
    raise ValueError("Can't perform bitwise operation on strings.")
  elif isinstance(A, str):
    return B
  elif isinstance(B, str):
    return A
  # XOR operation
  xor_result = np.bitwise_xor(A, B)

  # AND operation with A
  return np.bitwise_and(xor_result, A)


def get_final_mask(image_path, arm_or_palm='arm', smooth=True):
  body_mask = generate_body_mask(image_path)
  to_preserve_mask_image = to_preserve_mask(image_path)
  # feet_mask = feet_masking(image_path, smooth=smooth)    # -----> comment this one for getting foot as while
  final_mask = custom_bitwise_op(body_mask, to_preserve_mask_image)
  # final_mask = custom_bitwise_op(final_mask, feet_mask)    #  -----> comment this one for getting foot as while

  if arm_or_palm == 'arm':
    arm_mask = get_arm_mask(image_path)
    final_mask = custom_bitwise_op(final_mask, arm_mask)
  elif arm_or_palm == 'palm':
    palm_mask, _ = palm_masking(image_path)
    final_mask = custom_bitwise_op(final_mask, palm_mask)

  cloth_mask = generate_cloth_mask(image_path)
  final_mask = np.bitwise_or(final_mask, cloth_mask)
  return final_mask

def generate_final_mask(image_path):
    mask = get_final_mask(image_path, arm_or_palm='arm', smooth=True)
    output_path = f'{current_directory}/uploads/{str(uuid.uuid4())}.png'
    cv2.imwrite(output_path, mask)
    return output_path

@app.post("/generate-mask/")
async def mask_endpoint(base64_image: Base64Image):
    try:
        image_io = decode_base64_image(base64_image.base64_string)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())
        
        mask_path = generate_final_mask(image_path)

        with open(mask_path, "rb") as mask_file:
            mask_base64 = base64.b64encode(mask_file.read()).decode('utf-8')

        os.remove(image_path)
        os.remove(mask_path)

        return mask_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")

@app.post("/extract-cloth/")
async def cloth_extraction_endpoint(base64_image: Base64Image):
    try:
        image_io = decode_base64_image(base64_image.base64_string)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())
        
        extracted_cloth_path = main(image_path)

        with open(extracted_cloth_path, "rb") as mask_file:
            mask_base64 = base64.b64encode(mask_file.read()).decode('utf-8')

        os.remove(image_path)
        os.remove(extracted_cloth_path)

        return mask_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")


@app.post("/change-bg/")
async def bg_change_endpoint(images: List[Base64Image]):
    try:
        if len(images) != 2:
            raise HTTPException(status_code=400, detail="Two images are required")

        # Decode and save the original image
        org_image_io = decode_base64_image(images[0].base64_string)
        org_image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(org_image_path, "wb") as f:
            f.write(org_image_io.getvalue())

        # Decode and save the background image
        bg_image_io = decode_base64_image(images[1].base64_string)
        bg_image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(bg_image_path, "wb") as f:
            f.write(bg_image_io.getvalue())

        # Change the background
        changed_bg_path = change_bg(org_image_path, bg_image_path)

        # Encode the result to base64
        with open(changed_bg_path, "rb") as mask_file:
            changed_bg_base64 = base64.b64encode(mask_file.read()).decode('utf-8')

        # Cleanup
        os.remove(org_image_path)
        os.remove(bg_image_path)
        os.remove(changed_bg_path)

        return changed_bg_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")


@app.post("/perfect-change-bg/")
async def perfect_bg_change_endpoint(images: List[Base64Image]):
    try:
        if len(images) != 2:
            raise HTTPException(status_code=400, detail="Two images are required")

        # Decode and save the original image
        org_image_io = decode_base64_image(images[0].base64_string)
        org_image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(org_image_path, "wb") as f:
            f.write(org_image_io.getvalue())

        # Decode and save the background image
        bg_image_io = decode_base64_image(images[1].base64_string)
        bg_image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(bg_image_path, "wb") as f:
            f.write(bg_image_io.getvalue())

        # Change the background
        changed_bg_path = perfect_change_bg(org_image_path, bg_image_path)

        # Encode the result to base64
        with open(changed_bg_path, "rb") as mask_file:
            changed_bg_base64 = base64.b64encode(mask_file.read()).decode('utf-8')

        # Cleanup
        os.remove(org_image_path)
        os.remove(bg_image_path)
        os.remove(changed_bg_path)

        return changed_bg_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")

