# 🧍‍♂️ Human Detection using Computer Vision

This project implements a real-time human detection system using Python and OpenCV. It utilizes a pre-trained deep learning model to accurately detect humans in both live webcam streams and video files. The application is ideal for surveillance systems, automation, or safety monitoring.

## 📌 Features

- ✅ Real-time human detection via webcam or video file  
- ✅ Uses OpenCV's HOG (Histogram of Oriented Gradients) + SVM approach  
- ✅ Bounding box annotations for detected humans  
- ✅ Lightweight and fast performance on standard systems  
- ✅ Easy to extend with more advanced object detection models  

## 🧠 How It Works

This project uses:
- OpenCV’s `HOGDescriptor()` with a pre-trained people detector
- Frame-by-frame detection
- Real-time bounding boxes drawn on each detection

## 🛠️ Technologies Used

- Python  
- OpenCV  
- Numpy


## ▶️ Usage
🔴 To detect humans via webcam:
```bash
python human_detection.py
```
🎞️ To detect humans in a video file:
```bash
python human_detection.py --video sample_video.mp4
```


## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Karthi-1211/Human-Detection.git
cd Human-Detection
```

## 🚀 Future Improvements
Add YOLO or SSD model support for better accuracy

Implement face recognition or gesture detection

Log detections with timestamps

Integrate alarm or notification system
