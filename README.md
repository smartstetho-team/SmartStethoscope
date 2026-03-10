<div align="center">
  <img src="https://github.com/user-attachments/assets/731b596a-fda7-4439-a31a-851003d6ee57" width="200" height="200" alt="CardioScope Logo">

  # CardioScope
  **University of Waterloo | Mechatronics Engineering Capstone 2026**

  [![Website](https://img.shields.io/badge/Website-CardioScope-blue)](https://smartstetho-team.github.io/SmartStethoscope/)
  [![Status](https://img.shields.io/badge/Status-In--Development-green)](#)

  *An intelligent digital stethoscope designed for real-time heart sound analysis*
</div>

---

## 🩺 Project Overview
Cardiovascular disease is a leading cause of death globally, yet traditional acoustic stethoscopes lack the ability to record or algorithmically analyze heart sounds. **CardioScope** is a handheld device designed to bridge this gap by augmenting traditional auscultation with digital capabilities. 

Our system enables users to record and screen heart sounds remotely, utilizing a machine learning pipeline to flag potential irregularities like murmurs. By providing accessible, intelligent diagnostics, CardioScope assists users in determining when to seek professional medical care, particularly in low-resource or remote settings.

### 🚀 Key Objectives
* **Handheld Augmentation:** A compact (~200g), ergonomic device integrated with a custom-machined chest piece.
* **Intelligent Screening:** A digital pipeline targeting ≥60% sensitivity and specificity for detecting cardiac abnormalities.
* **Real-Time Connectivity:** Wireless streaming via Bluetooth Low Energy (BLE) to a companion React Native mobile app.
* **Performance Constraints:** Designed for a target build cost of ~$300 with at least one hour of continuous operation.

## 🛠️ Technical Stack
* **Core Hardware:** ESP32-S3 microcontroller, MAX4466 analog microphone, and a Waveshare 2-inch LCD for live visualization.
* **Power Management:** 3.7V LiPo battery paired with a PowerBoost 1000C module.
* **Software Framework:** Firmware written in C++ with FreeRTOS; mobile interface developed in React Native.
* **ML Pipeline:** Python-based digital filtering, feature extraction, and classification models for anomaly detection.

## 🖼️ Demonstration

<div align="center">
  <img src="https://github.com/user-attachments/assets/e4252bc6-475c-477e-ae77-177da68a3de2" width="40%" alt="CardioScope Final Prototype">

  ### 📺 Project Walkthrough
  [![CardioScope Demo Video](https://img.shields.io/badge/YouTube-Watch%20Demo-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=UnKrV0Ibpyw)
</div>

## 👥 The Team
We are final-year Mechatronics Engineering students at the University of Waterloo:
* **Krish Vijayan**
* **Om Patel**
* **Roby Aldave Garza**
* **Rijin Muralidharan**

---
