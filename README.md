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
  - Provides a safe exit mechanism using the `ESC` key.

## 3. Demonstration

#### Example of a Running Screen

This shows the program in action. You can see the detected lanes (green line), lane intrusion ROI (thin purple box), detected vehicles (green box), the tracked target vehicle (red box), the confirmed license plate (yellow text), and the plate view in the top-left corner.

<table>
  <tr>
    <td align="center"><img src="162313_D_v11.png" alt="v11.py Output image 1/2" width="auto" height="auto"></td>
    <td align="center"><img src="163614_D_v11.png" alt="v11.py Output image 2/2" width="auto" height="auto"></td>
  </tr>
</table>
<table>
  <tr>
    <td align="center"><img src="output_v11_162313_D.gif" alt="v11.py Output 1/2" width="auto" height="auto"></td>
    <td align="center"><img src="output_v11_163614_D.gif" alt="v11.py Output 2/2" width="auto" height="auto"></td>
  </tr>
</table>

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
  - The pre-trained model file (`EDSR_x4.pb`) used in this project was obtained from the [Saafke/EDSR_Tensorflow](https://github.com/Saafke/EDSR_Tensorflow) repository, which is licensed under the **Apache License 2.0**. The full text of the license is included in this repository.[Apache 2.0 License](LICENSE_of_EDSR_x4)
* **Libraries:**
  * **OpenCV:** https://opencv.org
  * **PyTorch:** https://pytorch.org
  * **EasyOCR:** https://github.com/JaidedAI/EasyOCR
* **AI Assistant:**
  * This project was developed in collaboration with AI Assistant **Google Gemini**. Gemini provided assistance with idea materialization, algorithm design, code debugging, implementation of advanced techniques (Super-Resolution, trackers, voting systems), and `README.md` generation.

## 7. Limitations & Future Work

While this project successfully builds a system for detecting lane-intruding vehicles and recognizing their license plates, several areas for improvement remain to address all variables of a real-world road environment and to achieve higher accuracy.
In the future, I aim to address these challenges by implementing the methods detailed below, as well as developing a lightweight AI model specifically for Korean license plate identification.

### 7-1. General Areas for Improvement

#### 7-1-1. Physical Limitations of Low-Resolution & Long-Distance Plate Recognition
- **Limitation:** The current system enhances the recognition rate of low-resolution plates through Super-Resolution and multi-frame analysis techniques. However, there is a fundamental limit to restoring license plates that are too far away in the original video to be identifiable.
- **Future Work:** I intend to consider using higher-resolution cameras or hardware with optical zoom capabilities, or setting a confidence range to only attempt recognition on vehicles within a certain distance.

#### 7-1-2. Trade-off between Real-time Processing Speed and Accuracy
- **Limitation:** The current system improves efficiency with a hybrid approach, using the CPU for vehicle detection and the GPU for license plate detection/recognition. However, executing all functions at peak performance on every single frame remains computationally expensive.
- **Future Work:** I believe a higher FPS could be achieved by introducing model quantization or pruning techniques, or by running the vehicle detection model with PyTorch/TensorRT instead of `cv2.dnn` to process the entire pipeline on the GPU.

#### 7-1-3. Generalization Performance for Diverse Environments
- **Limitation:** The system's current parameters are tuned for the provided video data (daytime, clear weather). Performance may degrade under different lighting and weather conditions, such as at night, during rain, or in fog.
- **Future Work:** I am considering acquiring additional data from diverse environments and implementing more robust preprocessing techniques (e.g., contrast enhancement specialized for night-time videos), or developing a system to manage different parameter sets for each condition.

#### 7-1-4. Advanced Tracking Algorithm
- **Limitation:** The current CSRT tracker initiates a new search when it loses a target. It can be vulnerable to long-term occlusion scenarios, such as when a vehicle is completely hidden by another car.
- **Future Work:** To improve stability, instead of simply resetting upon tracking failure, a more intelligent tracking logic could be implemented by combining the tracker with a Kalman Filter to predict the vehicle's next position and re-identify it after a temporary disappearance.

### 7-2. Critiques of Specific Outputs

#### 7-2-1. v11.py Output 1/2
- **Limitation:** In the current result, the system only recognizes that the target vehicle has crossed the lane a certain amount of time after the lane change occurred.
- **Future Work:** To enable faster intrusion detection, the criteria could be expanded to include the vehicle's entire bottom edge within the recognition scope, rather than just the center point of the bottom edge. However, this must be considered in conjunction with model lightening to manage the computational load.

#### 7-2-2. v11.py Output 2/2
- **Limitation:** It is not possible to identify the license plate number of the vehicle.
- **Future Work:** It will be necessary to consider creating an upscaling and image enhancement model that exceeds EDSR_x4, or a process that can identify the plate by stacking the maximum possible number of frames.

## 8. License

This project is licensed under the [MIT License](LICENSE).
