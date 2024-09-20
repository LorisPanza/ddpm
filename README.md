# DDPM: Denoising Diffusion Probabilistic Model

This repository contains an implementation of a **Denoising Diffusion Probabilistic Model (DDPM)**. The DDPM model is a generative model designed to iteratively improve noisy images over a series of timesteps, eventually generating a high-quality image from random noise.

# Introduction

The DDPM is a novel generative model based on the idea of modeling the forward process of adding noise to data and then learning the reverse process that denoises the data back to the original distribution. This repository implements the complete training and evaluation pipeline for a DDPM model, demonstrating how to train the model on a given dataset and generate new data by reversing the diffusion process.

Key paper: **[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)** by Ho et al.

# Features

- Full implementation of the DDPM forward and reverse processes.
- Configurable model architecture and training parameters.
- Supports training on custom datasets.
- Generates high-quality samples by reversing the diffusion process.
- Visualization tools for tracking training progress and sample generation.
