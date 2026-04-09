# 🧠 EEG2Vote: A Multimodal Dataset for Detecting Social Judgments from Brain Activity


**EEG2Vote** is a large-scale multimodal dataset and machine learning benchmark suite designed to decode the implicit social biases and subjective leadership preferences of individuals directly from neural signals. 

## Table of Contents
- [Overview](#-overview)
- [The Three Main Components of EEG2Vote](#-the-three-pillars-of-eeg2vote)
- [Dataset Array Structure](#-dataset-array-structure-eegvote_singlesubnpz)
- [Quick Start: Loading the Data](#-quick-start-loading-the-data)
- [Running the Benchmarks](#-running-the-benchmarks)
- [Citation](#-citation)

---

## Overview
Traditional affective computing relies heavily on explicit behavioral feedback, which is vulnerable to social desirability bias. EEG2Vote bridges this semantic gap by capturing the continuous, millisecond-resolution brain responses of **38 participants** as they implicitly evaluate the "leadership" qualities of photorealistic human faces.

**Key Features:**
* **Stimuli:** 1,160 highly controlled facial prototypes generated via StyleGAN3 ($z \in \mathbb{R}^{512}$).
* **Hardware:** Synchronized 64-channel BioSemi ActiveTwo EEG (1024 Hz).
* **Paradigms:** Rapid Serial Visual Presentation (RSVP) coupled with explicit behavioral ground-truth voting.
* **Tasks:** Binary classification (0 = Non-Leader, 1 = Leader).
* **Scale:** 73,479 total paired trials.

---

## The Three Main Components of EEG2Vote
To maximize accessibility for cognitive neuroscientists, computer vision experts, and behavioral psychologists, the preprocessed data is hosted via **[Google Drive](https://drive.google.com/drive/folders/18gVJZP-H299PuMsJClqoyGoCygRtQtqz?usp=drive_link)** in ready-to-train formats. Download the files and place them in the `./data/` directory.

### 1. The EEG Dataset
*
EEG_datasets/
├── **`eeg2vote_singlesub.npz`:** Single-subject isolated datasets for individualized cognitive modeling.
├── **`eeg2vote_crosssub.npz`:** Global dataset fused for cross-subject generalization tasks.
└── **`eegdata(set).zip`:** Raw `.set` EEGLAB files for researchers wishing to perform custom artifact rejection, filtering, or epoching.

* Example Dataset Array Structure (`eegvote_singlesub.npz`) *

To accommodate varying channel counts after individualized artifact rejection, the EEG data are stored in a subject-specific dictionary format. For every subject `[ID]`, the following arrays are provided:

| Array Key Format | Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `Sub_[ID]_X` | `(n_trials, n_channels, 1025)` | `float32` | Continuous EEG epochs. *Note: `n_channels` varies slightly per subject due to bad channel removal (e.g., 53 to 60). `1025` represents the timepoints.* |
| `Sub_[ID]_latent` | `(n_trials, 512)` | `float32` | The 512-dimensional generative facial latent vector corresponding to the visual stimulus shown in each trial. |
| `Sub_[ID]_y` | `(n_trials,)` | `int64` | Binary leadership vote (0 = Not Chosen, 1 = Chosen). |
| `Sub_[ID]_ch_names`| `(n_channels,)` | `str` | The specific 10-20 EEG channel labels retained for that subject. |

---

### 2. The Vision Dataset
*
EEG2Vote_vision_dataset/
├── subject_trials_labels.csv # Trial IDs, subject IDs, and vote labels
├── images/                  # Folder for .jpg image files (e.g., 443.jpg)
└── eeg2vote_visual_data.npz

### 3. The Behavioral Dataset
* **behavior_voting/ (Folder)**: Contains explicit post-experiment Likert-scale evaluations of candidate Competence, Trustworthiness, and Likeability, alongside Implicit Association Test (IAT) scores and subject demographics.
* 
behavioral_voting/
├── demographics/
│   └── subject_demographics.csv    # Subject age and sex
├── iat/
│   ├── IAT_react.csv               # Mean response ordinals for IAT questions
│   ├── IAT_Questionnaires.csv       # IAT questionnaire items and count stats
│   ├── IAT_Questionnaires_rating.csv # Detailed prejudice and sexism ratings
│   ├── IAT_stats.csv               # Mean response times and IAT scores per subject
│   └── IATAllTrials.csv            # Raw trial-level data for IAT leadership experiment
└── voting/
    ├── face_annotations.csv        # Face ID, age/sex metadata, and subject voting data
    └── feedbackdata                # Competence, likeability, and trustworthiness scores
---


## Quick Start: Loading the Data

### Experiment Setup
git clone https://github.com/Affective-Neural-Interaction-Group/EEG2Vote/
cd EEG2Vote
pip install -r requirements.txt

```python
import numpy as np
import pandas as pd

# Load the Single-Subject Multimodal Archive
data = np.load('./data/eegvote_singlesub.npz')

# Extract paired multimodal data for Subject 13
eeg_features = data['Sub_13_X']         # Shape: (1966, 60, 1025)
visual_latents = data['Sub_13_latent']  # Shape: (1966, 512)
labels = data['Sub_13_y']               # Shape: (1966,)
channels = data['Sub_13_ch_names']      # Shape: (60,)

print(f"Subject 13 loaded: {eeg_features.shape[0]} trials ready for training.")
```

## Running the Benchmarks
