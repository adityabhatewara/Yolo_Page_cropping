# 📜 Ancient Script Digitization: YOLO Segmentation & Auto-Alignment

An automated computer vision pipeline designed to streamline the archival process of ancient manuscripts. This project uses a custom-trained YOLO segmentation model to detect, extract, and geometrically straighten photographs of historical documents, replacing a highly tedious manual editing workflow.

## 💡 The Problem & Business Value
Archival companies photograph thousands of ancient scripts and hard copies to preserve them digitally. However, these raw photographs are often skewed, taken at an angle, or surrounded by noisy backgrounds. Previously, each image had to be manually cropped and rotated to be readable. 

This project fully automates that preprocessing pipeline, saving countless hours of manual data entry and ensuring standardized, perfectly flat digital archives.

## 🧠 The Extraction Pipeline
This system doesn't just draw a box; it understands the exact contours of the physical paper.

1. **Instance Segmentation (YOLO):** The raw photograph is fed into a YOLO segmentation model trained specifically on document edges, ignoring complex backgrounds and isolating the script's exact pixel mask.
2. **Contour & Corner Extraction:** The system processes the segmentation mask to identify the four primary corners of the physical document.
3. **Geometric Transformation (OpenCV):** Using the extracted coordinates, a Warp Perspective transformation is applied. This mathematically "flattens" the image, correcting any camera skew and rotating the script so the text is perfectly horizontal.

## 📸 Pipeline Visualized


### 1. Original vs. YOLO Segmentation Mask
![Segmentation Step](assets/segmentation.png)
*The model accurately identifies the boundaries of the script despite the background.*

### 2. The Final Output: Auto-Straightened & Cropped
![Final Result](assets/final_result.png)
*The skewed, raw photo is geometrically transformed into a clean, flat scan ready for the archive.*

## 🛠️ Tech Stack
* **Deep Learning:** Ultralytics YOLO (Instance Segmentation)
* **Computer Vision:** OpenCV (Contour detection, Warp Perspective, Auto-rotation)
* **Language:** Python 3.x

## 🚀 How to Run
1. Clone the repository and install dependencies:
   ```bash
   git clone [https://github.com/adityabhatewara/Yolo_Page_cropping.git](https://github.com/adityabhatewara/Yolo_Page_cropping.git)
   pip install -r requirements.txt
