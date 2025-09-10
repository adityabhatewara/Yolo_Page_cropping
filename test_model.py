import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# 1. Load YOLO model
# -----------------------------
model_path = 'runs/segment/train/weights/best.pt'  # path to your trained model
model = YOLO(model_path)

# -----------------------------
# 2. Load a sample image
# -----------------------------
image_path = 'sample.jpg'  # replace with your test image
image = cv2.imread(image_path)
overlay = image.copy()

# -----------------------------
# 3. Run inference
# -----------------------------
results = model.predict(source=image_path, save=False, verbose=False)[0]

if len(results.masks.data) == 0:
    print("No mask detected!")
else:
    mask = results.masks.data[0].cpu().numpy()  # binary mask
    mask_uint8 = (mask * 255).astype(np.uint8)

    # -----------------------------
    # 4. Draw mask in semi-transparent red
    # -----------------------------
    overlay[mask == 1] = [0, 0, 255]  # red shading
    alpha = 0.5
    shaded = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)

    # -----------------------------
    # 5. Approximate quadrilateral
    # -----------------------------
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    # Draw rectangle on top of mask
    cv2.drawContours(shaded, [box], 0, (0, 255, 0), 2)  # green rectangle

    # -----------------------------
    # 6. Show the final output
    # -----------------------------
    cv2.imshow("Mask + Quadrilateral", shaded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
