import cv2
import mediapipe as mp
import numpy as np
import os

def detect_and_crop_faces_with_shoulders_fixed_ratio(input_folder, output_folder, output_size=(128, 192), expand_ratio=0.5):
    """
    Detect and crop face regions including shoulders, maintaining a vertical rectangular ratio.
    
    Args:
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str): Path to the folder to save output images.
    - output_size (tuple): Target size of the cropped image (width, height).
    - expand_ratio (float): Ratio to expand the bounding box.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load OpenCV's pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        if not os.path.isfile(filepath):
            continue

        # Read the image
        img = cv2.imread(filepath)
        if img is None:
            print(f"Cannot read file: {filename}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            print(f"No face detected: {filename}")
            continue

        # Process the first detected face
        for (x, y, w, h) in faces[:1]:
            # Expand the bounding box to include shoulders
            x_expanded = max(0, int(x - expand_ratio * w))
            y_expanded = max(0, int(y - expand_ratio * h))
            w_expanded = int(w * (1 + 2 * expand_ratio))
            h_expanded = int(h * (1 + expand_ratio * 1.5))  # Expand height more to include shoulders
            
            # Ensure the cropping area is within the image bounds
            x_expanded_end = min(img.shape[1], x_expanded + w_expanded)
            y_expanded_end = min(img.shape[0], y_expanded + h_expanded)

            # Crop the expanded region
            cropped_face = img[y_expanded:y_expanded_end, x_expanded:x_expanded_end]
            
            # Ensure the cropped region matches the target vertical rectangle aspect ratio
            crop_h, crop_w = cropped_face.shape[:2]
            target_aspect_ratio = output_size[1] / output_size[0]  # Height-to-width ratio

            if crop_h / crop_w > target_aspect_ratio:  # If too tall, crop top and bottom
                new_h = int(crop_w * target_aspect_ratio)
                y_center = crop_h // 2
                cropped_face = cropped_face[y_center - new_h // 2:y_center + new_h // 2, :]
            else:  # If too wide, crop left and right
                new_w = int(crop_h / target_aspect_ratio)
                x_center = crop_w // 2
                cropped_face = cropped_face[:, x_center - new_w // 2:x_center + new_w // 2]
            
            # Save the cropped image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_face)
            print(f"Processed: {filename}")
            break


def replace_background(input_folder, output_folder, background_path):
    """
    Replace the background of images, supporting both transparent PNG and non-transparent images.
    
    Args:
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str): Path to the folder to save output images.
    - background_path (str): Path to the virtual background image.
    """
    mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the background image exists
    if not os.path.exists(background_path):
        raise FileNotFoundError(f"Background image not found: {background_path}")

    # Load the background image
    background = cv2.imread(background_path)

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        if not os.path.isfile(filepath):
            continue

        # Read the input image
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # Support for transparent PNG

        if img is None:
            print(f"Unable to read file: {filename}")
            continue

        # Check if the image has a transparent background (PNG with alpha channel)
        if img.shape[2] == 4:  # Check for alpha channel
            print(f"Transparent background PNG detected: {filename}")
            alpha = img[:, :, 3] / 255.0  # Extract alpha channel
            img_rgb = img[:, :, :3]
            bg_resized = cv2.resize(background, (img_rgb.shape[1], img_rgb.shape[0]))

            # Composite the background
            final_image = bg_resized.copy()
            for c in range(3):  # Loop through RGB channels
                final_image[:, :, c] = img_rgb[:, :, c] * alpha + bg_resized[:, :, c] * (1 - alpha)
        else:
            print(f"Non-transparent image, using Mediapipe for segmentation: {filename}")

            # Resize background image to match input image dimensions
            bg_resized = cv2.resize(background, (img.shape[1], img.shape[0]))

            # Convert to RGB (required by Mediapipe)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Segment the foreground
            results = mp_selfie_segmentation.process(img_rgb)
            mask = results.segmentation_mask > 0.1  # Set segmentation threshold

            # Convert the mask to uint8
            mask = (mask.astype(np.uint8) * 255)

            # Apply morphological operations to refine mask edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_refined = cv2.dilate(mask, kernel, iterations=1)  # Dilation
            mask_refined = cv2.erode(mask_refined, kernel, iterations=1)  # Erosion

            # Blur mask edges for smoother transitions
            mask_blurred = cv2.GaussianBlur(mask_refined, (5, 5), 0)

            # Replace background
            final_image = bg_resized.copy()
            for c in range(3):  # Loop through RGB channels
                final_image[:, :, c] = img[:, :, c] * (mask_blurred / 255) + bg_resized[:, :, c] * (1 - (mask_blurred / 255))

        # Save the result
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, final_image)
        print(f"Background replacement completed: {filename}")



if __name__ == "__main__":
    # Paths to input and output folders
    input_folder = "input_images"
    output_folder = "output_images"
    
    # Target size
    output_size = (128, 192)  # Width 128, height 192, vertical rectangular ratio
    
    # Expansion ratio
    expand_ratio = 1.5  # Expand 50%, adjustable as needed
    
    detect_and_crop_faces_with_shoulders_fixed_ratio(input_folder, output_folder, output_size, expand_ratio)
# Paths to input and output folders
    input_folder = "output_images"
    output_folder = "output1_images"

    # Path to the virtual background
    background_path = "backg.png"

    # Execute background replacement
    replace_background(input_folder, output_folder, background_path)
