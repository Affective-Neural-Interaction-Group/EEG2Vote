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


conda create -n brainvote python=3.8
conda activate brainvote
pip install -r requirements.txt
