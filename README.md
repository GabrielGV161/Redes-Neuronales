# 🫀 SNN-Arrhythmia-Detector: Neuromorphic ECG Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Brian2](https://img.shields.io/badge/Simulator-Brian2-green)
![Status](https://img.shields.io/badge/Status-Clinical%20Validation-orange)

A bio-inspired **Spiking Neural Network (SNN)** designed to detect cardiac arrhythmias using **Spike-Timing-Dependent Plasticity (STDP)**. This project bridges the gap between biological signal processing and neuromorphic computing, validating its performance against real-world data from the **MIT-BIH Arrhythmia Database**.

---

## 🔬 Scientific Abstract

Standard ECG analysis relies on digital signal processing or heavy deep learning models (CNN/RNN). This project explores a **neuromorphic approach**, encoding analog ECG signals into discrete spike trains. The network employs a competitive **Dual-Population Architecture** with lateral inhibition, where neurons specialize in detecting either synchronous (Healthy) or asynchronous (Arrhythmic) patterns based on the morphological jitter of the QRS complex.

## 🚀 Key Features

* **Dual-Tau Architecture:** Implements two competing neuronal populations with distinct membrane time constants ($\tau$):
    * **Healthy Team (Low $\tau$):** Acts as a coincidence detector for precise, rhythmic signals.
    * **Arrhythmia Team (High $\tau$):** Acts as a temporal integrator for chaotic or wide QRS complexes.
* **Unsupervised Learning (STDP):** The network learns to distinguish patterns without explicit labeling during the training phase, utilizing biological Hebbian learning rules.
* **Winner-Take-All Competition:** Implements strong lateral inhibition to force decision-making between the diagnostic populations.
* **Clinical Validation:** Integrated directly with the **MIT-BIH Database** (PhysioNet) to test against real patient records (e.g., Patient 115 vs. Patient 203).

## 📂 Project Structure

```text
├── src/
│   ├── main_ecg.py          # Entry point: Orchestrates Training & Validation
│   ├── network.py           # SNN Architecture Definition (LIF Neurons + STDP)
│   ├── real_ecg_loader.py   # Signal Processing: Analog ECG -> Spike Encoding
│   ├── weight_manager.py    # Persistence Layer (Save/Load Synaptic Weights)
│   ├── test_recognition.py  # Inference Engine & Diagnostic Logic
│   └── visualization.py     # Plotting tools for Membrane Potentials
├── saved_weights/           # Serialized network state (.pkl)
├── requirements.txt         # Dependencies
└── README.md
   
