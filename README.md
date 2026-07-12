# Halm-AI-Powered-Helmet-for-Enhanced-Worker-Protection

**AI-Powered Helmet for Enhanced Worker Protection**

Halm is an intelligent safety system designed to bridge the gap between traditional protective equipment and proactive industrial safety. By integrating Computer Vision and Kinematic Analysis into a single wearable platform, the system shifts safety management from reactive mitigation to real-time, preventative intervention.

---

## Overview

The Halm project addresses the functional fragmentation found in current industrial safety solutions. It utilizes an edge-processing architecture (Raspberry Pi 5) to autonomously identify environmental hazards and detect worker falls, delivering instant situational awareness to a centralized dashboard.

This project was developed as a Bachelor of Science graduation project in Computer Science at King Faisal University.

---

## Key Features

* **Real-time Spatial Hazard Localization:** Uses an edge-optimized YOLOv8 model to identify pits, sharp tools, and unsafe areas.
* **Smart Sense Kinematic Detection:** Employs a BNO085 IMU and dual-threshold algorithms for instantaneous fall detection.
* **Centralized Data Analytics:** Fuses vision warnings and kinetic data into a unified dashboard for informed, rapid intervention.
* **Instant Local Alerting:** Triggers a local LED warning on the helmet to immediately notify the worker of detected risks.

---

## Technologies Used

* **AI / Computer Vision:** YOLOv8
* **Programming:** Python
* **Edge Computing:** Raspberry Pi 5
* **Dashboard:** React + Vite (Frontend) and Flask (Backend)
* **Hardware Sensors:** BNO085 IMU and Garmin GPS

---

## Repository Structure

Repository Structure
HALM-Smart-Safety-Helmet/
├── Halm_Website/
│   ├── backend/          # Flask API and SQLite database
│   └── src/              # React Dashboard source code
├── rpi_scripts/          # Hardware implementation scripts (Vision, Fall, GPS)
├── model/                # AI development & evaluation notebooks
└── README.md             # Project documentation

---

## Project Context

This system was developed as a Graduation Project in partial fulfillment of the requirements for the degree of Bachelor of Science in Computer Science at King Faisal University.

**Students:**

* Hissah Almuhaysh
* Latifah Alhafith
* Rana Althafar
* Ftoon Althafar

---

## Acknowledgments

* **Supervised by:** Dr. Hala Hamdun
* **Committee Members:** Dr. Rawabi Alsedais

Special thanks to our families and mentors for their unwavering encouragement throughout this research process.


