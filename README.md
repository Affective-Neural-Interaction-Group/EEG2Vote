# 🧠 EEG2Vote: A multimodal dataset for detecting social judgements from brain activity

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

**EEG-Vote** is a multimodal dataset and machine learning benchmark suite designed to decode implicit social biases of individuals. 



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
* **Tasks:** Binary classification (Leader vs. Non-Leader)

---

## 💾 Dataset Access & Formats
To maximize accessibility for both cognitive neuroscientists and computer vision/ML researchers, the preprocessed data is hosted on **https://drive.google.com/drive/folders/18gVJZP-H299PuMsJClqoyGoCygRtQtqz?usp=drive_link** in ready-to-train NumPy archives (`.npz`) and EEGLAB data files (`.set`). 

Download the `.npz` files and place them in the `./data/` directory.

1. `eegvote_crosssub.npz`: Global cross-subject dataset with binary labels (0 = Non-Leader, 1 = Leader).
2. `eegvote_singlesub.npz`: Single-subject dataset with binary labels (0 = Non-Leader, 1 = Leader).
3. `visual_dataset.npz`: Visual stmuli dataset with binary labels (0 = Non-Leader, 1 = Leader)
4. `eegdata(set).zip`: Global cross-subject dataset with binary labels (0 = Non-Leader, 1 = Leader)
*(Note: Raw `.set` EEGLAB files are also available for researchers wishing to perform custom artifact rejection).*

---

## Dataset Overview (`eegvote_singlesub.npz`)

**Quick Facts:**
* **Total Trials:** 73,479 paired observations
* **Unique Subjects:** 38 participants
* **Trials per Subject:** Average of ~ 1933.66 ± 130.53 SD (Range: 1288 - 2079)
* **Task:** Binary leadership election (0 = Not Chosen, 1 = Chosen)
* **Modalities:** Continuous EEG 

### Array Structure
To accommodate varying channel counts after artifact rejection, the EEG data is stored in a subject-specific dictionary format. The generative visual features (StyleGAN latents) are paired directly with these trials.

For every subject `[ID]`, the following arrays are provided:

| Array Key Format | Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `Sub_[ID]_X` | `(n_trials, n_channels, 1025)` | `float32` | Continuous EEG data for the subject. Note: `n_channels` varies slightly per subject due to bad channel removal (e.g., 53 to 60 channels). `1025` represents the timepoints. |
| `Sub_[ID]_latent` | `(n_trials, 512)` | `float32` | The 512-dimensional generative facial latent vector corresponding to the visual stimulus shown in each trial. |
| `Sub_[ID]_y` | `(n_trials,)` | `int64` | Binary leadership vote (0 = Not Chosen, 1 = Chosen). |
| `Sub_[ID]_ch_names`| `(n_channels,)` | `String` | The specific 10-20 EEG channel labels retained for that subject. |

**Loading Example (Python):**
```python
import numpy as np

# Load the archive
data = np.load('eegvote_singlesub.npz')

# Extract paired multimodal data for Subject 13
eeg_features = data['Sub_13_X']         # Shape: (1966, 60, 1025)
visual_latents = data['Sub_13_latent']  # Shape: (1966, 512)
labels = data['Sub_13_y']               # Shape: (1966,)

## 📂 Repository Structure

```text
eeg2vote/
│
├── data/                                 # Place downloaded .npz files here
│   ├── eegvote_singlesub.npz
│   ├── eegvote_crosssub.npz
│   └── visual_dataset.npz
│
├── scripts_preprocessing/                # MNE-Python pipeline scripts
│   ├── 01_make_global_dataset.py
│   └── 02_make_single_subject_dataset.py
│
├── scripts_training/                     # PyTorch & Braindecode baselines
│   ├── train_dl.py   # EEGNet, ShallowFBCSPNet, Conformer, BIOT
│   └── train_traditional_ml.py           # CSP+LDA, CSP+RF, PCA+SVM
│
├── requirements.txt                      # Python dependencies
└── README.md
