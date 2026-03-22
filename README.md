# 🧠 Brainvote: A Multimodal EEG Dataset for Implicit Social Bias

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

**Brainvote** is a large-scale, multimodal dataset and deep learning benchmark suite designed to decode and synthesize implicit social biases using neurophysiological signals. 

This repository contains the preprocessing scripts, machine learning baselines (CNNs, Transformers, and Traditional ML), and data-loading utilities for the official Brainvote dataset, as detailed in our ACM Multimedia paper: *"[Insert Your Paper Title Here]"*.

## 📑 Table of Contents
- [Overview](#-overview)
- [Dataset Access & Formats](#-dataset-access--formats)
- [Repository Structure](#-repository-structure)
- [Quick Start: Loading the Data](#-quick-start-loading-the-data)
- [Running the Benchmarks](#-running-the-benchmarks)
- [Environment Setup](#-environment-setup)
- [Ethical Use and License](#-ethical-use-and-license)
- [Citation](#-citation)

---

## 🔍 Overview
Traditional affective computing relies heavily on explicit behavioral feedback, which is vulnerable to social desirability bias. Brainvote bridges this semantic gap by capturing the continuous, millisecond-resolution brain responses of **38 participants** as they implicitly evaluate the "leadership" qualities of photorealistic human faces.

**Key Features:**
* **Stimuli:** 1,160 highly controlled facial prototypes generated via StyleGAN3 ($z \in \mathbb{R}^{512}$).
* **Hardware:** Synchronized 64-channel BioSemi ActiveTwo EEG (1024 Hz).
* **Paradigms:** Rapid Serial Visual Presentation (RSVP) coupled with explicit behavioral ground-truth voting.
* **Tasks:** Binary classification (Leader vs. Non-Leader), 4-class preference intensity, and within-subject personalization.

---

## 💾 Dataset Access & Formats
To maximize accessibility for both cognitive neuroscientists and computer vision/ML researchers, the preprocessed data is hosted on **[Insert Zenodo/PhysioNet Link Here]** in ready-to-train NumPy archives (`.npz`). 

Download the `.npz` files and place them in the `./data/` directory.

1. `brainvote_2class_ready.npz`: Global cross-subject dataset with binary labels (0 = Non-Leader, 1 = Leader).
2. `brainvote_4class_ready.npz`: Global cross-subject dataset with ordinal intensity labels (0 to 3).
3. `brainvote_within_subject_max_chans.npz`: Dynamically sized matrices preserving the absolute maximum number of clean channels for individual subject training.

*(Note: Raw `.set`/`.fdt` EEGLAB files are also available upon request for researchers wishing to perform custom artifact rejection).*

---

## 📂 Repository Structure

```text
brainvote/
│
├── data/                                 # Place downloaded .npz files here
│   ├── brainvote_2class_ready.npz
│   └── brainvote_within_subject_max_chans.npz
│
├── scripts_preprocessing/                # MNE-Python pipeline scripts
│   ├── 01_make_global_dataset.py
│   └── 02_make_within_subject_dataset.py
│
├── scripts_training/                     # PyTorch & Braindecode baselines
│   ├── train_global_cnn.py
│   ├── train_within_subject_unified.py   # EEGNet, ShallowFBCSPNet, Conformer, BIOT
│   └── train_traditional_ml.py           # CSP+LDA, CSP+RF, PCA+SVM
│
├── requirements.txt                      # Python dependencies
└── README.md
