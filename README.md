<div align="right">

**[한국어](./README.ko.md)** | **[English](./README.md)**

</div>

# Real-time Lane Intrusion Detection and License Plate Recognition System

## 1. Introduction

This project is a computer vision system designed to automatically detect vehicles intruding into the current lane from blackbox footage, recognize their license plates, extract the text, and record it with a timestamp.

The primary focus was not just on implementing individual features, but on building a hybrid pipeline that combines multiple deep learning models and advanced image processing techniques to overcome real-world challenges such as low resolution, lighting changes, and various object angles and distances.

## 2. Key Features

- **Stable Lane Detection:**
  - Detects lane lines based on OpenCV's Hough Transform.
  - Implements **Temporal Smoothing** using `collections.deque` to minimize jittering of lane lines caused by vehicle movement to ensure stable detection.

- **Hybrid Deep Learning Model Pipeline:**
  - **Vehicle Detection (CPU-based):** Utilizes a lightweight `YOLOv3-tiny` model running on the CPU for fast initial vehicle detection, ensuring system responsiveness.
  - **License Plate Localization (GPU-based):** Employs a specialized **YOLOv5** model, accelerated on the GPU, to accurately find the precise location of license plates within the detected vehicle area.
  - **Character Recognition (GPU-based):** Upgraded from traditional Tesseract OCR to the deep-learning-based **EasyOCR** library, running on the GPU for significantly improved accuracy and speed.

- **Intelligent License Plate Processing:**
  - **Automatic Plate Correction:** Even if the detected plate is tilted, **Perspective Transform** is used to automatically correct it into a straight, rectangular image, improving OCR input quality.
  - **Super-Resolution (SR):** Implements an **EDSR** model to enhance the quality of small, blurry license plate images from distant vehicles before they are fed into the OCR engine.
  - **Temporal Voting System:** A character-level voting mechanism consolidates unstable OCR results from multiple frames to confirm a single, high-confidence license plate number, ensuring result reliability.

- **Advanced Object Tracking:**
  - Integrates the **CSRT Tracker** from OpenCV to stably follow a target vehicle once it has been identified as intruding into the lane.
  - The tracker works in tandem with the YOLO vehicle detector, which periodically corrects the tracker's position, leading to efficient and robust tracking.

- **User Experience Features:**
  - Saves the final processed video with all annotations to an `.avi` file.
  - Allows real-time playback speed control using keyboard arrow keys (↑, ↓).
  - Provides a safe exit mechanism using the `ESC` key.

## 3. Demonstration

#### Example of a Running Screen

This shows the program in action. You can see the detected lanes (green line), detected vehicles (green box), the tracked target vehicle (red box), the confirmed license plate (yellow text), and the plate view in the top-left corner.

![v11.py Output 1/2](output_v11_162313_D.gif)[v11.py Output 2/2](output_v11_163614_D.gif)


#### Sample Log Output (`log.txt`)

When a license plate is confirmed, it is recorded with a timestamp in `log.txt`.

```
번호판: 64가1511 (대상 차량 크기: 298x258)
```

## 4. System Architecture

This system operates on the following pipeline structure:

```
[Input Video Frame]
      |
      +-----> 1. Lane Detection (OpenCV, CPU) & Temporal Smoothing
      |
      +-----> 2. Vehicle Detection (YOLOv3-tiny, CPU)
      |
      +-----> 3. Lane Intrusion Check (IoU) & Start/Update Tracking (CSRT Tracker)
                |
                +-----> 4. License Plate Localization (YOLOv5, GPU)
                          |
                          +-----> 5. (Optional) Super-Resolution for quality enhancement (GPU)
                                    |
                                    +-----> 6. Character Recognition (EasyOCR, GPU)
                                              |
                                              +-----> 7. Confirm Final Plate via Temporal Voting
                                                        |
                                                        +-----> [Output Result & Log]
```

## 5. Installation & How to Run

#### Prerequisites
* **Python 3.10**
* **NVIDIA GPU(Check for GPU support for the CUDA Toolkit version below)**
* **CUDA Toolkit 12.1** & compatible cuDNN

#### Environment Setup
This project was tested in a **Python 3.10 virtual environment**.

1.  **Create and activate a virtual environment:**
    ```bash
    py -3.10 -m venv venv
    venv\Scripts\activate
    ```
2.  **Install required libraries:**
    ```bash
    # Install PyTorch for GPU (for CUDA 12.1)
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

    # Install YOLOv5 and EasyOCR
    pip install yolov5 easyocr

    # Install standard OpenCV with contrib modules
    pip install opencv-contrib-python
    ```
3.  **Download Model Files:**
    Download the following files and place them in the project root directory.
    * **Vehicle Detector:** `yolov3-tiny.weights`, `yolov3-tiny.cfg`, `coco.names`
    * **License Plate Detector:** `lp_det.pt` (or `best.pt`) - from a repository like [sauce-git/korean-license-plate-detector](https://github.com/sauce-git/korean-license-plate-detector)
    * **Super-Resolution Model:** `EDSR_x4.pb` - (e.g., from [Saafke/EDSR_Tensorflow](https://github.com/Saafke/EDSR_Tensorflow/tree/master/models))

#### Running the Script

Execute the following command in your terminal from the project folder.
```bash
python your_script_name.py
```
- Press the `ESC` or `q` key to exit the program safely. The processed video and log file will be generated.

## 6. References & Open Source

This project was built upon the ideas and resources from the following open sources.

* **Models:**
  * **YOLOv3 / YOLOv3-tiny:** Joseph Redmon, Ali Farhadi ([Paper](https://pjreddie.com/darknet/yolo/))
  * **YOLOv5:** Ultralytics ([GitHub](https://github.com/ultralytics/yolov5))
  * **Korean License Plate Detector Model:** [sauce-git/korean-license-plate-detector](https://github.com/sauce-git/korean-license-plate-detector)
  * **EDSR Super-Resolution Model:** Bee Lim, Sanghyun Son et al. ([Paper](https://arxiv.org/abs/1707.02921))
  - The pre-trained model file (`EDSR_x4.pb`) used in this project was obtained from the [Saafke/EDSR_Tensorflow](https://github.com/Saafke/EDSR_Tensorflow) repository, which is licensed under the **Apache License 2.0**. The full text of the license is included in this repository.
* **Libraries:**
  * **OpenCV:** https://opencv.org
  * **PyTorch:** https://pytorch.org
  * **EasyOCR:** https://github.com/JaidedAI/EasyOCR
* **AI Assistant:**
  * This project was developed in collaboration with AI Assistant **Google Gemini**. Gemini provided assistance with idea materialization, algorithm design, code debugging, implementation of advanced techniques (Super-Resolution, trackers, voting systems), and `README.md` generation.

## 7. License

This project is licensed under the [MIT License](LICENSE).
