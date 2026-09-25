GlucoAI 🩸📱

GlucoAI is an Android application that uses an on-device machine learning model to estimate blood glucose levels from PPG (Photoplethysmography) signals.

The project is designed as a lightweight, offline-first prototype that performs signal processing and ML inference directly on an Android device without requiring a cloud backend.

Note: GlucoAI is an academic/project prototype and is not a medical device. Its predictions should not be used for diagnosis, treatment, medication, or other medical decisions.

🚀 Features

📱 Native Android application

📷 PPG signal acquisition using the smartphone camera

❤️ PPG signal quality checking

📊 Signal preprocessing and feature extraction

🤖 On-device TensorFlow Lite glucose prediction

🔌 Offline inference — no internet connection is required for ML prediction

⚡ Lightweight mobile inference

🧪 Error handling for unusable/noisy PPG signals

🎨 Simple Android UI for capturing a signal and displaying the prediction

🧠 How It Works

The basic pipeline is:

Finger placed over camera
        ↓
Camera captures PPG signal
        ↓
RGB signal extraction
        ↓
PPG preprocessing
        ↓
Signal quality check
        ↓
Feature / signal preparation
        ↓
TensorFlow Lite model
        ↓
Glucose prediction
        ↓
Result displayed in Android app

The trained ML model is converted to TensorFlow Lite and bundled inside the Android application.

Because the .tflite model is stored locally in the application assets, prediction can be performed without sending the user's signal to a remote server.

🏗️ Project Structure

GlucoAI-App/
│
├── app/
│   └── src/
│       └── main/
│           ├── java/com/example/glucoai/
│           │   ├── MainActivity.kt
│           │   ├── GlucosePredictor.kt
│           │   └── PPGProcessor.kt
│           │
│           ├── assets/
│           │   └── glucoai_mobile.tflite
│           │
│           ├── res/
│           │   ├── drawable/
│           │   ├── layout/
│           │   ├── mipmap/
│           │   └── values/
│           │
│           └── AndroidManifest.xml
│
├── gradle/
├── build.gradle.kts
├── settings.gradle.kts
├── gradle.properties
├── gradlew
├── gradlew.bat
└── .gitignore

🛠️ Tech Stack

Component

Technology

Platform

Android

Language

Kotlin

IDE

Android Studio

Build System

Gradle

ML Framework

TensorFlow Lite

Signal

PPG

Input

RGB camera signal

Inference

On-device

Model Format

.tflite

🤖 Machine Learning

The project uses a trained glucose prediction model that has been converted to TensorFlow Lite for mobile deployment.

The mobile application loads the model from:

app/src/main/assets/glucoai_mobile.tflite

The Android application uses TensorFlow Lite's Interpreter to perform inference locally.

Conceptually:

val interpreter = Interpreter(model)

The application then prepares the processed PPG signal and passes it to the model to obtain the glucose prediction.

📱 Running the Project

Requirements

Android Studio

Android SDK

JDK compatible with the project's Gradle/Android Gradle Plugin configuration

Android phone or emulator

Camera permission for PPG acquisition

Setup

Clone the repository:

git clone https://github.com/harris1230/GlucoAI-App.git

Open the project in Android Studio.

Allow Android Studio to sync the Gradle project.

Connect an Android device or start an emulator.

Grant the required camera permission.

Click:

Run ▶

🔐 Privacy

GlucoAI is designed so that ML inference can happen locally on the Android device.

The application does not need a cloud API to run the prediction model.

This means:

PPG Signal
    ↓
Android Device
    ↓
Local Processing
    ↓
Local TFLite Model
    ↓
Prediction

No server is required for the core prediction workflow.

👥 Team Collaboration

The project uses Git and GitHub for collaboration.

Recommended workflow:

main
 │
 ├── feature/ui
 ├── feature/camera
 ├── feature/ml
 └── feature/testing

Team members should:

Create a feature branch.

Make their changes.

Commit their changes.

Push the branch to GitHub.

Open a Pull Request.

Review and merge the changes into main.

Example

git checkout -b feature/camera

git add .
git commit -m "Improve camera PPG capture"

git push origin feature/camera

Then create a Pull Request on GitHub.

📦 APK Generation

To generate an APK from Android Studio:

Build
  → Build Bundle(s) / APK(s)
  → Build APK(s)

For a release build, configure your signing credentials locally.

Do not commit private signing keys (.jks / .keystore) to GitHub.

⚠️ Limitations

This project is a research/academic prototype.

Factors such as:

Camera hardware

Lighting conditions

Finger placement

Motion

Skin/contact conditions

PPG signal quality

Training-data limitations

can affect the prediction.

The model should therefore not be considered a replacement for a clinically validated glucose meter or continuous glucose monitor.

🔬 Project Goals

The main goals of GlucoAI are:

Explore non-invasive glucose estimation using PPG signals.

Apply machine learning to physiological signals.

Convert an ML model into a mobile-friendly format.

Perform ML inference directly on Android.

Build a practical end-to-end ML + Android application.

👨‍💻 Development

The main Android components are:

MainActivity.kt

Controls the application UI and overall prediction workflow.

PPGProcessor.kt

Handles PPG signal processing and signal-quality-related operations.

GlucosePredictor.kt

Loads the TensorFlow Lite model and performs on-device inference.

glucoai_mobile.tflite

The trained TensorFlow Lite model used for glucose prediction.

📄 License

This project is intended for academic and educational purposes.

Add an appropriate open-source license if the project is later intended for public redistribution.

👤 Author

Harish Gupta

B.Tech Computer Engineering

GlucoAI — Android + Machine Learning Project.
