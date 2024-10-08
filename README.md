

---

# Facial Emotion Detector

## Overview

This project is a real-time **Facial Emotion Detector** that uses a convolutional neural network (CNN) to recognize human emotions through a webcam. It detects seven basic emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The project uses OpenCV for face detection and Keras for emotion classification.

## Features

- Detects faces using OpenCV's Haar Cascade Classifier.
- Recognizes emotions in real-time from webcam footage.
- Classifies emotions into seven categories.

## Requirements

Install the necessary libraries before running the code:

```bash
pip install opencv-python
pip install keras
pip install tensorflow
pip install numpy
```

## Project Files

- **facialemotionmodel.h5**: Pre-trained Keras model for emotion recognition.
- **realtimedetection.py**: Script for real-time emotion detection.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Face_Emotion_By_Amanteja.git
   cd Face_Emotion_By_Amanteja
   ```

2. Load the pre-trained model:
   Make sure `facialemotionmodel.h5` is in the folder.

3. Run the script:
   ```bash
   python realtimedetection.py
   ```

   The webcam will open, and faces detected in the video will be classified into emotions. 

4. Exit:
   Press the `Esc` key to close the webcam and stop the program.

![Untitled](https://github.com/user-attachments/assets/08cfa968-e057-48ae-99d7-eb823c9dcfad)


## Notes

- The webcam index is set to `cv2.VideoCapture(0)`. If it doesn't work, you might need to change the index to match your system.
- If there are any encoding issues, check your environment’s text encoding settings.

## Acknowledgments

- **FER2013 dataset**: Used for training the model.
- **Keras** and **TensorFlow**: For building the deep learning model.
- **OpenCV**: For face detection using Haar Cascades.

---

