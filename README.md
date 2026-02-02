This project is an AI-based real-time facial analysis system developed using Python, OpenCV, and deep learning models, integrated with a Flask web application. The system captures live video through a webcam, detects a human face, and analyzes multiple facial attributes such as emotion, age group, smile status, and mask usage.

The application uses computer vision techniques like Haar Cascade classifiers for face and smile detection, a pre-trained CNN model for age estimation, and an ONNX-based deep learning model for mask detection. Emotion recognition is performed using a feature-based facial analysis approach, ensuring stable and real-time performance without frequent misclassification.

Based on the detected emotion, the system provides emotion-aware responses:

Displays a motivational joke when sadness is detected

Recommends a peaceful song when anger is detected

Sends an email alert to a guardian when fear is detected

After detecting the face and emotion, the camera automatically captures an image, closes the video stream, and displays the results on an interactive web interface.
