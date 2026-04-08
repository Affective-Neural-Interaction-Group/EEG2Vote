# 🧠 EEG2Vote: A Multimodal Dataset for Detecting Social Judgments from Brain Activity


**EEG2Vote** is a large-scale multimodal dataset and machine learning benchmark suite designed to decode the implicit social biases and subjective leadership preferences of individuals directly from neural signals. 

## Table of Contents
- [Overview](#-overview)
- [The Three Pillars of EEG2Vote](#-the-three-pillars-of-eeg2vote)
- [Dataset Array Structure](#-dataset-array-structure-eegvote_singlesubnpz)
- [Repository Structure](#-repository-structure)
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

## The Three Components of EEG2Vote
To maximize accessibility for cognitive neuroscientists, computer vision experts, and behavioral psychologists, the preprocessed data is hosted via **[Google Drive](https://drive.google.com/drive/folders/18gVJZP-H299PuMsJClqoyGoCygRtQtqz?usp=drive_link)** in ready-to-train formats. Download the files and place them in the `./data/` directory.

### 1. The Neurophysiological (EEG) Dataset
* **`eegvote_singlesub.npz`:** Single-subject isolated datasets for individualized cognitive modeling.
* **`eegvote_crosssub.npz`:** Global dataset fused for cross-subject generalization tasks.
* **`eegdata(set).zip`:** Raw `.set` EEGLAB files for researchers wishing to perform custom artifact rejection, filtering, or epoching.

### 2. The Vision Dataset
* **`eegvote_visual_dataset.npz`:** A standalone Computer Vision benchmark containing the visual stimuli and their corresponding binary leadership labels. Visual stimulus images are paired with these trials, allowing for pure deep-learning facial analysis without requiring EEG expertise.

### 3. The Behavioral & Implicit Dataset
* **behavior_voting/ (Folder)**: Contains explicit post-experiment Likert-scale evaluations of candidate Competence, Trustworthiness, and Likeability, alongside Implicit Association Test (IAT) scores and subject demographics.

---

## Dataset Array Structure (`eegvote_singlesub.npz`)

To accommodate varying channel counts after individualized artifact rejection, the EEG data is stored in a subject-specific dictionary format. For every subject `[ID]`, the following arrays are provided:

| Array Key Format | Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `Sub_[ID]_X` | `(n_trials, n_channels, 1025)` | `float32` | Continuous EEG epochs. *Note: `n_channels` varies slightly per subject due to bad channel removal (e.g., 53 to 60). `1025` represents the timepoints.* |
| `Sub_[ID]_latent` | `(n_trials, 512)` | `float32` | The 512-dimensional generative facial latent vector corresponding to the visual stimulus shown in each trial. |
| `Sub_[ID]_y` | `(n_trials,)` | `int64` | Binary leadership vote (0 = Not Chosen, 1 = Chosen). |
| `Sub_[ID]_ch_names`| `(n_channels,)` | `str` | The specific 10-20 EEG channel labels retained for that subject. |

---

## Quick Start: Loading the Data

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
