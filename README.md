# BrainVote: Neural Decoding of Voting Intent from EEG

This repository contains the official implementation, data processing pipeline, and analysis scripts for the paper:
**"BrainVote: Reconstructing Face Images from Neural Voting Intentions"**

## 📄 Abstract

BrainVote is a Brain-Computer Interface (BCI) framework capable of decoding explicit voting intentions from EEG signals evoked by face stimuli. By utilizing a multi-view ensemble of EEGNet models (targeting Delta, Beta, Broadband, and Late-Latency activity), we achieved a classification accuracy of **71.7%** and an AUC of **0.73**, significantly outperforming standard broadband baselines (p<.001).

This repository provides the tools to reproduce the decoding results and statistical analyses reported in the study.

## 🛠️ Installation

### Clone the repository
```bash
git clone [https://github.com/yourusername/brainvote.git](https://github.com/yourusername/brainvote.git)
cd brainvote
```

### Install dependencies
We recommend using a Conda environment to manage dependencies.
```bash
conda create -n brainvote python=3.8
conda activate brainvote
pip install -r requirements.txt
```

## 📂 Data Format
Due to storage size limits, the full raw dataset is hosted externally. [Insert Link to OSF/Zenodo dataset here]

However, the scripts in this repository expect input data tensors of the following shape:

Input Shape: (Epochs, Channels, Timepoints)

Sampling Rate: 250 Hz (Downsampled from 1000 Hz)

Epoch Length: 1000 ms (-200ms to +800ms)

## 🚀 Usage
1. Preprocessing (MATLAB/EEGLAB)
The raw EEG data is preprocessed using the script located at scripts/01_preprocess.m.

Filters: 1–80 Hz Bandpass, 45–55 Hz Notch.

Artifacts: ICA-based rejection (EOG/EMG).

Output: Cleaned .set files or .npy arrays ready for Python training.

2. Training the Ensemble
To train the 4 constituent models (Broadband, Delta, Beta, Late) and generate the results, run:
```bash
python scripts/02_train_models.py --subject_id all --epochs 80 --batch_size 32
```
Model: EEGNet (Lawhern et al., 2018)

Configuration: 5-fold stratified cross-validation.

Output: Saves trained metrics to results/metrics/voting_eegnet_final_detailed.pkl.

Expected Output:

Delta vs Beta: Significant performance difference (p<.05).

Ensemble vs Broadband: Significant improvement (p<.001).

Global Pooled AUC: 0.73.
