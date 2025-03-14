# DDPM Implementation from Scratch
This repository contains a PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) built from scratch. DDPMs are a class of generative models that learn to generate images by gradually denoising a random Gaussian noise.
## Overview
Denoising Diffusion Probabilistic Models work by:

- Forward process - gradually adding noise to images according to a schedule
- Training a neural network to predict the noise in noisy images
- Reverse process (sampling) - iteratively denoising random noise to generate new images

This implementation follows the methodology described in the original paper *"Denoising Diffusion Probabilistic Models" by Ho et al*.
## Repository Structure
```shell
├── ddpm.py           # Core DDPM model implementation
├── main.py           # Training and visualization script
├── dataset.py        # Dataset loading and preprocessing
└── README.md         # This file
```

## Features

Complete DDPM implementation with configurable parameters
Linear noise schedule as described in the original paper
U-Net architecture with time embeddings for noise prediction
Training pipeline with visualization tools
Step-by-step sampling process visualization

## Usage
### Training
To train the model:
```shell
python training.py
```
This will:

1. Load the dataset (default is MNIST with images normalized to [-1, 1])
2. Initialize the DDPM model
3. Train the model using the diffusion training process
4. Visualize original, noised, and reconstructed samples

## Citation
```shell
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2006.11239},
  year={2020}
}
```
