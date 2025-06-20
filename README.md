# AI Drowsiness Detection System

## 🎯 Overview

The **AI Drowsiness Detection System** is a sophisticated real-time monitoring application that uses computer vision and machine learning to detect signs of drowsiness. Built with a modern, beautiful interface, it provides immediate alerts for safety-critical applications like driver monitoring, workplace safety, and healthcare monitoring.

This project demonstrates a complete machine learning pipeline from data collection to deployment, featuring:
- **Real-time drowsiness detection** using dual YOLOv8 models
- **Modern PyQt5 interface** with beautiful, responsive design
- **Automated data labeling** using GroundingDINO
- **Comprehensive training pipeline** with detailed analytics

---

## 🎥 Demo

Watch the real-time driver drowsiness detection demo in action:

[![Watch the demo](https://img.youtube.com/vi/ynLUTRclpUc/0.jpg)](https://youtu.be/ynLUTRclpUc?si=gmcy-rB7S_dNzV-9)

> Click the thumbnail above or follow this link: [https://youtu.be/ynLUTRclpUc?si=gmcy-rB7S_dNzV-9](https://youtu.be/ynLUTRclpUc?si=gmcy-rB7S_dNzV-9)


---

## Group member

- **Ziwei Zhou** — Designing and training model
- **Zuqing Meng** — Collecting and pre-processing data 
- **Zitong He** — Finding data
- **Yuxin Wang** — Finding data


---

## 📊 Model & Dataset & Data Source

The model weights are included in the folder **"./model/eye_detect"** and **"./model/yawn_detect"**

The training data for this project is available through the following source:

### **📁 Dataset Download**: 
https://drive.google.com/file/d/1ZRrk3Sk-dTqiYTAQ-zdoAf_LQRxjAEKb/view?usp=sharing

This dataset contains:
- **Eye Detection Images**: Labeled images for open and closed eye classification
- **Yawn Detection Images**: Labeled images for yawn detection
- **Training Data**: Organized training and validation sets
- **Annotations**: Bounding box coordinates and class labels

To use this dataset:
1. Download the dataset from the Google Drive link above
2. Extract the files to your project directory
3. Update the `dataset.yaml` configuration file with the correct paths
4. Run the training script: `python train_customize.py`

### **📁 Source of the Data**
Kaggle: https://www.kaggle.com/datasets/cubeai/drowsiness-detection-for-yolov8

HuggingFace: https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes


---

## ✨ Key Features

### 🎨 **Modern User Interface**
- **Beautiful Design**: Professional gradient backgrounds, rounded corners, and modern typography
- **Real-time Monitoring**: Live camera feed with instant status updates
- **Interactive Dashboard**: Status indicators with progress bars and color-coded alerts
- **Smart Alerts**: Dynamic alert banners with different severity levels
- **Responsive Layout**: Adapts to different screen sizes and resolutions

### 🤖 **AI-Powered Detection**
- **Dual Model System**: Separate YOLOv8 models for eye closure and yawning detection
- **Facial Landmark Tracking**: MediaPipe integration for precise facial feature detection
- **Real-time Processing**: Multi-threaded architecture for smooth performance
- **Confidence-based Detection**: Intelligent threshold management for accurate results

### 📊 **Comprehensive Analytics**
- **Live Statistics**: Blink count, microsleep duration, yawn frequency, and duration
- **Visual Progress Indicators**: Color-coded progress bars (green → yellow → red)
- **Status Monitoring**: Context-aware status messages with color coding
- **Performance Metrics**: Training results, confusion matrices, and evaluation curves

---

## 📁 Project Structure

```
drowsiness_detect/
├── 📱 Main Applications
│   └── drowsy_detector.py            # Main detection application with PyQt5 interface
│
├── 🛠️ Data Processing Tools
│   ├── labelling.py                  # Automated bounding box labeling
│   ├── capture_data_images.py        # Image data collection utility
│   ├── LoadData.py                   # Dataset loading and preprocessing
│   └── RedirectData.py               # Data organization and redirection
│
├── 🎓 Training & Models
│   ├── train_customize.py            # Custom YOLO model training script
│   ├── dataset.yaml                  # Dataset configuration
│   └── model/                        # Trained model weights
│       ├── eye_detect/               # Eye detection model
│       │   ├── best.pt               # Best eye detection model
│       │   └── last.pt               # Latest eye detection model
│       └── yawn_detect/              # Yawn detection model
│           ├── best.pt               # Best yawn detection model
│           └── last.pt               # Latest yawn detection model
│
├── 📋 Configuration & Documentation
│   ├── requirements.txt              # Python dependencies
│   └── README.md                     # This documentation
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Webcam or video input device
- 4GB+ RAM recommended

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd drowsiness_detect
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python drowsy_detector.py
   ```

---

## 🎮 Usage Guide

### Main Application (`drowsy_detector.py`)

The main application features a modern two-panel interface:

#### Left Panel - Live Camera Feed
- **Real-time video display** with professional styling
- **Facial landmark tracking** for precise detection
- **Automatic ROI extraction** for eyes and mouth regions

#### Right Panel - Monitoring Dashboard
- **Status Indicators**: Four interactive panels showing:
  - 👁️ **Blinks**: Total blink count with progress bar
  - 💤 **Microsleeps**: Duration with color-coded alerts
  - 😮 **Yawns**: Frequency tracking
  - ⏳ **Yawn Duration**: Prolonged yawn detection

- **Alert Banner**: Dynamic alerts for:
  - ⚠️ **Warning**: Prolonged yawn detected (>2 seconds)
  - 🚨 **Danger**: Microsleep risk detected (>2 seconds)

- **Status Messages**: Context-aware feedback:
  - ✅ **Active**: System monitoring normally
  - 😮 **Yawn**: Yawn detected - stay alert
  - 😴 **Risk**: Eyes closed - microsleep risk

### Data Collection Tools

#### `capture_data_images.py`
Records image data for training:
```bash
python capture_data_images.py
```

#### `labelling.py`
Automates bounding box generation using GroundingDINO:
```bash
python labelling.py
```

### Training Pipeline

#### `train_customize.py`

---

## 🧠 Technical Architecture

### Detection Pipeline

1. **Video Capture**: Multi-threaded frame capture from webcam
2. **Face Detection**: MediaPipe FaceMesh for facial landmark extraction
3. **ROI Extraction**: Automatic region-of-interest detection for eyes and mouth
4. **Model Inference**: Dual YOLOv8 models for classification
5. **State Tracking**: Temporal analysis for blink and yawn detection
6. **Alert Generation**: Threshold-based alert system
7. **UI Updates**: Real-time interface updates with visual feedback

### Model Specifications

#### Eye Detection Model
- **Architecture**: YOLOv8 classification
- **Classes**: Open Eye (0), Close Eye (1)
- **Model Location**: `model/eye_detect/best.pt`
- **Performance**: High accuracy for real-time detection

#### Yawn Detection Model
- **Architecture**: YOLOv8 classification
- **Classes**: Yawn (0), No Yawn (1)
- **Model Location**: `model/yawn_detect/best.pt`
- **Performance**: Robust detection with confidence thresholds

### Interface Components

#### VideoThread Class
```python
class VideoThread(QThread):
    - Multi-threaded video processing
    - Real-time frame analysis
    - State tracking and statistics
    - Alert signal generation
```

#### DrowsinessDetector Class
```python
class DrowsinessDetector(QMainWindow):
    - Main application window
    - Video display panel
    - Statistics dashboard
    - Alert management system
```

---

## 🎨 Design System

### Color Palette
- **Primary Blue**: #007bff (buttons, highlights)
- **Success Green**: #28a745 (safe conditions)
- **Warning Yellow**: #ffc107 (caution conditions)
- **Danger Red**: #dc3545 (alert conditions)
- **Neutral Gray**: #6c757d (secondary elements)
- **Background**: Gradient from #f8f9fa to #e9ecef

### Typography
- **Primary Font**: Segoe UI, Arial, sans-serif
- **Hierarchy**: 24px (titles), 18px (subtitles), 14px (body), 12px (captions)
- **Weight**: Bold for headings, regular for body text

### Layout Principles
- **Two-panel design** for clear information separation
- **Grid-based layout** for consistent spacing
- **Responsive elements** that adapt to screen size
- **Visual hierarchy** with proper contrast and sizing

---

## 📈 Performance & Optimization

### Real-time Processing
- **Multi-threading**: Separate threads for capture and processing
- **Queue Management**: Efficient frame buffering
- **Memory Management**: Proper resource cleanup
- **UI Responsiveness**: Non-blocking interface updates

### Model Optimization
- **Confidence Thresholds**: Optimized for accuracy vs speed
- **ROI Processing**: Focused analysis on relevant regions
- **Batch Processing**: Efficient inference pipeline
- **Model Quantization**: Optimized model weights

---

## 🔧 Configuration

### Dataset Configuration (`dataset.yaml`)
```yaml
path: datasets
train: ./runs/detecteye
val: images/val
nc: 2
names:
  0: open eye
  1: close eye
```

### Model Parameters
- **Detection Confidence**: 0.30 for eyes, 0.50 for yawns
- **Alert Thresholds**: 2s for yawns, 2s for microsleeps
- **Frame Rate**: 45ms processing interval
- **Queue Size**: 2 frames for smooth processing

---

## 🐛 Troubleshooting

### Common Issues

1. **Camera Not Found**
   ```bash
   # Check camera permissions
   # Ensure no other application is using the camera
   ```

2. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

3. **Model Loading Errors**
   ```bash
   # Verify model weights exist
   ls model/eye_detect/
   ls model/yawn_detect/
   ```

4. **Performance Issues**
   - Reduce video resolution
   - Close other applications
   - Check system resources

---

## Run the application
Run with verbose output:
```bash
python drowsy_detector.py --debug
```

---

## 📊 Training Results

### Eye Detection Model Performance
- **Accuracy**: High precision for both open and closed eyes
- **Model Weights**: Available in `model/eye_detect/`
- **Training Script**: `train_customize.py`

### Yawn Detection Model Performance
- **Accuracy**: Robust detection across different conditions
- **Model Weights**: Available in `model/yawn_detect/`
- **Performance Metrics**: Precision, recall, and F1-score

---

## 🔮 Future Enhancements

### Planned Features
1. **Dark Mode**: Toggle between light and dark themes
2. **Customizable Thresholds**: User-adjustable alert levels
3. **Data Export**: Export monitoring data to CSV/JSON
4. **Settings Panel**: Configuration options for the interface
5. **Multi-person Detection**: Support for multiple subjects
6. **Mobile Deployment**: iOS/Android app versions
7. **Cloud Integration**: Remote monitoring capabilities
8. **Advanced Analytics**: Machine learning insights and trends

### Technical Improvements
- **Model Optimization**: Quantization and pruning
- **Performance Tuning**: GPU acceleration support
- **Accessibility**: Screen reader support and keyboard navigation
- **Internationalization**: Multi-language support

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests if applicable**
5. **Submit a pull request**

### Development Setup
```bash
# Clone and setup development environment
git clone <repository-url>
cd drowsiness_detect
pip install -r requirements.txt
python drowsy_detector.py  # Test the application
```

---

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the detection framework
- **MediaPipe**: Google for facial landmark detection
- **PyQt5**: Qt Company for the GUI framework
- **OpenCV**: Intel for computer vision capabilities
- **GroundingDINO**: Microsoft for automated labeling

---



**Note**: This system is designed for educational and research purposes. For production use in safety-critical applications, additional validation and testing is recommended.