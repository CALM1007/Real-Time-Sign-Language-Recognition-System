# Real-Time Sign Language Recognition System

This project is a real-time sign language recognition system developed to help improve communication accessibility for users who rely on sign language. The system uses a webcam to capture live video input, detects the hand region using MediaPipe, and classifies the detected gesture using a trained Convolutional Neural Network (CNN) model.

The recognized ASL gestures are converted into readable text output through a sentence reconstruction function. A word suggestion feature is also included to help users complete words more efficiently. The prototype is implemented as a web-based application using Flask, with OpenCV for video processing, MediaPipe for hand detection, TensorFlow for deep learning inference, and HTML/CSS/JavaScript for the user interface.

## Key Features

* Real-time webcam-based hand gesture recognition
* MediaPipe hand detection and hand region cropping
* CNN-based ASL gesture classification
* Recognition of ASL alphabet and special classes such as space, delete, and nothing
* Sentence reconstruction from recognized gestures
* Word suggestion for faster text formation
* Confidence score and FPS display
* Web-based interface using Flask

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* Flask
* HTML, CSS, JavaScript
