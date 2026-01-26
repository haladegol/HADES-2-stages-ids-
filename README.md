HADES: Hierarchical Intrusion Detection System
A two-stage intelligent IDS that quickly filters normal traffic and accurately classifies malicious attacks for efficient network security.

📌 Overview
HADES (Hierarchical Intrusion Detection System) is a resource-efficient cybersecurity solution designed to solve the "Efficiency Dilemma" in modern high-speed networks.

Traditional Intrusion Detection Systems (IDS) often struggle between two extremes:

Speed vs. Accuracy: Fast systems (signature-based) miss zero-day threats, while accurate systems (deep learning) are too slow for real-time traffic.

Resource Waste: Analyzing safe traffic (which is the majority of network flow) with heavy AI models wastes valuable CPU and RAM.

HADES implements a "Binary-to-Multiclass" architecture to solve this. It splits detection into two stages: a lightweight "Gatekeeper" that rapidly filters out safe traffic, and a deep "Analyst" that investigates only the suspicious packets.

🏗️ Architecture
The system operates on a Filter-and-Analyze pipeline:

Stage 1: The Gatekeeper (Binary Classification)
Model: Random Forest (Scikit-learn)

Goal: Ultra-fast filtering.

Function: Scans network flows to answer a simple question: "Is this safe or suspicious?"

Outcome: * Benign Traffic: Immediately allowed through (discarded from analysis pipeline).

Suspicious Traffic: Forwarded to Stage 2.

Key Metric: High Recall (Zero-tolerance for false negatives).

Stage 2: The Analyst (Multiclass Classification)
Model: XGBoost

Goal: High-fidelity attack identification.

Function: Performs deep analysis on the filtered "suspicious" traffic to identify the specific attack vector (e.g., DoS, Botnet, Web Attack).

Outcome: Detailed alert generation and MITRE ATT&CK mapping.

✨ Key Features
Resource Efficiency: Optimizes CPU and Memory usage by discarding benign traffic early.

Zero-Budget Deployment: Designed to run effectively on a single standard workstation without expensive cloud infrastructure or proprietary hardware.

Hybrid Intelligence: Combines the speed of standard ML with the accuracy of boosting algorithms.

Streamlit Dashboard: Interactive real-time monitoring interface for Security Analysts.

Standard Compliance: Maps detected threats to MITRE ATT&CK global standards.

🛠️ Technology Stack
Language: Python 3.9+

Machine Learning: * Scikit-learn (Random Forest)

XGBoost (Gradient Boosting)

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Dashboarding: Streamlit

Database: SQLite (Lightweight logging)

Dataset: CSE-CIC-IDS2018 (Flow-based intrusion detection dataset)

🚀 Getting Started
Prerequisites
Python 3.9 or higher

Virtual Environment (recommended)

Installation
Clone the repository

Bash

git clone https://github.com/yourusername/HADES-IDS.git
cd HADES-IDS
Create a virtual environment

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

Bash

pip install -r requirements.txt
Run the Dashboard

Bash

streamlit run app.py
📊 Results
Speed: Stage 1 (Gatekeeper) operates significantly faster than deep analysis, processing the majority of traffic with minimal latency.

Accuracy: Stage 2 (Analyst) achieves high precision in classifying complex attack patterns under class imbalance.

Hardware: Successfully validated on standard laptops (16GB RAM, i7 CPU) without crashing.

👥 Team & Contributors
This project was developed as a Thesis for the Arab Academy for Science, Technology, and Maritime Transport (Smart Village) under the supervision of Dr. Ahmed Maher.

Hala Degol

Hagar Mahmoud Fathy Soliman

Hagar Ahmed Abosamra

Abdulkarim Mustafa Byloneh

Osama Zaid

🔮 Future Work
Real-Time Defense: Evolution from offline prototype to inline blocking.

Enterprise Integration: Connection to SOCs and SIEM platforms.

Scalability: Migration to cloud infrastructure for high-volume traffic handling.

Automated Retraining: Pipelines to learn from new zero-day threats automatically.
