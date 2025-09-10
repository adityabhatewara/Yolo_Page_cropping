'''import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# 1. Load trained YOLOv11 model
# -----------------------------
model_path = 'runs/segment/train/weights/best.pt'  # path to your trained model
model = YOLO(model_path)

# -----------------------------
# 2. Load sample image
# -----------------------------
image_path = 'sample.jpg'  # replace with your image
image = cv2.imread(image_path)

# -----------------------------
# 3. Run inference
# -----------------------------
results = model.predict(source=image_path, save=False, verbose=False)[0]  # first image result
mask = results.masks.data[0].cpu().numpy()  # binary mask (H x W)

# -----------------------------
# 4. Find largest contour of the mask
# -----------------------------
mask_uint8 = (mask * 255).astype(np.uint8)
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

# -----------------------------
# 5. Minimum area rectangle (to get rotation angle)
# -----------------------------
rect = cv2.minAreaRect(contour)  # returns ((cx,cy), (w,h), angle)
box = cv2.boxPoints(rect)        # 4 corner points
box = box.astype(int)

# -----------------------------
# 6. Rotate image to make rectangle horizontal
# -----------------------------
angle = rect[2]
if rect[1][0] < rect[1][1]:  # width < height → adjust angle
    angle = angle - 90

# rotation matrix
center = rect[0]
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

# rotate mask as well
rotated_mask = cv2.warpAffine(mask_uint8, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

# -----------------------------
# 7. Crop the rotated rectangle
# -----------------------------
# find contours again on rotated mask
contours_rot, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_rot = max(contours_rot, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(contour_rot)
cropped = rotated[y:y+h, x:x+w]

# -----------------------------
# 8. Show results
# -----------------------------
cv2.imshow('Rotated & Cropped Quadrilateral', cropped)  
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import cv2
import numpy as np
import os
from ultralytics import YOLO

# -----------------------------
# 1. User configuration
# -----------------------------
model_path = 'runs/segment/train/weights/best.pt'  # your trained YOLO model
input_folder = 'input_images'  # folder containing images to process
output_folder = 'cropped_pages'  # folder to save aligned cropped pages

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# 2. Load YOLO model
# -----------------------------
model = YOLO(model_path)

# -----------------------------
# 3. Process images
# -----------------------------
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for idx, image_name in enumerate(image_files, start=1):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # -----------------------------
    # 3a. Run YOLO segmentation
    # -----------------------------
    results = model.predict(source=image_path, save=False, verbose=False)[0]

    if len(results.masks.data) == 0:
        print(f"No mask detected for {image_name}, skipping.")
        continue

    mask = results.masks.data[0].cpu().numpy()  # binary mask
    mask_uint8 = (mask * 255).astype(np.uint8)

    # -----------------------------
    # 3b. Find largest contour
    # -----------------------------
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # -----------------------------
    # 3c. Minimum area rectangle
    # -----------------------------
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    if rect[1][0] < rect[1][1]:
        angle -= 90
    center = rect[0]

    # -----------------------------
    # 3d. Rotate image and mask
    # -----------------------------
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(orig_image, M, (orig_image.shape[1], orig_image.shape[0]), flags=cv2.INTER_CUBIC)
    rotated_mask = cv2.warpAffine(mask_uint8, M, (orig_image.shape[1], orig_image.shape[0]), flags=cv2.INTER_NEAREST)

    # -----------------------------
    # 3e. Crop the aligned rectangle
    # -----------------------------
    contours_rot, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_rot = max(contours_rot, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour_rot)
    cropped = rotated_image[y:y+h, x:x+w]

    # -----------------------------
    # 3f. Save the result
    # -----------------------------
    output_name = f"page_{idx}.jpg"
    output_path = os.path.join(output_folder, output_name)
    cv2.imwrite(output_path, cropped)
    print(f"Saved {output_name}")
