# Import of libraries
import numpy as np


import torch
import torch.nn as nn


import math

from unet import Unet


class DDPM(nn.Module):
    def __init__(self, timesteps, imagechannels=3, channels=64, time_embedding_size=16):
        # Initialize the DDPM model with configurable parameters
        # timesteps: Total number of diffusion steps
        # imagechannels: Number of input image channels (default: 3 for RGB)
        # channels: Base channel count for U-Net (default: 64)
        # time_embedding_size: Dimension of time step embeddings (default: 16)
        
        super(DDPM, self).__init__()
        self.timesteps = timesteps
        
        # Define beta schedule parameters as mentioned in DDPM papers
        # Linear schedule from min=0.0001 to max=0.02
        self.max = 0.02
        self.min = 0.0001
        
        # Create noise schedule arrays
        self.betas = np.linspace(self.min, self.max, timesteps)  # Linear beta schedule
        self.alphas = 1 - self.betas  # Alpha values (1 - beta)
        self.alphas_cumsum = self.alphas.cumprod()  # Cumulative product of alphas
        self.sqrt_alpha_cum = np.sqrt(self.alphas_cumsum)  # Square root of cumulative alphas
        
        # Initialize U-Net model with time conditioning
        self.unet = Unet(image_channels=imagechannels, channels=channels, time_embedding_size=time_embedding_size, ch_mults=(1,2,2))
        self.loss_mse = nn.MSELoss()  # Mean squared error loss function
    
    def forward(self, x, t):
        # Forward pass: simply passes input to U-Net with timestep
        # x: Input images
        # t: Timestep information
        return self.unet(x, t)
    
    def training_process(self, batch):
        # Main training function implementing the DDPM training algorithm
        
        # Sample random timesteps for each image in the batch
        ts = torch.randint(low=0, high=self.timesteps, size=[batch.shape[0]])
        
        # Generate random noise for the batch
        noise = torch.randn(batch.shape)
        noised_images = []
        
        # Apply forward diffusion process: Add noise according to timestep schedule
        for i in range(len(ts)):
            sqrt_alpha_cum_batched = self.sqrt_alpha_cum[ts[i]]
            alpha_cum_batched = self.alphas_cumsum[ts[i]]
            # Forward diffusion formula: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
            noised_image = batch[i] * (sqrt_alpha_cum_batched) + math.sqrt((1-alpha_cum_batched)) * noise[i]
            noised_images.append(noised_image)
        
        noised_images = torch.stack(noised_images, dim=0)
        
        # Use model to predict the noise that was added (denoising step)
        predicted_noise = self.forward(noised_images, ts)
        
        # Loss is MSE between actual noise and predicted noise
        loss_error_estimation = self.loss_mse(noise, predicted_noise)
        
        return loss_error_estimation
    
    def sampling(self, x, t):
        # Sampling function for the reverse diffusion process
        # Implements a single reverse diffusion step from x_t to x_{t-1}
        
        with torch.no_grad():
            # Only add noise during intermediate steps (not at t=0)
            if t > 1:
                z = torch.randn(x.shape)  # Random noise for stochasticity
            else:
                z = 0  # No noise for final step
            
            # Predict noise in the current sample
            predicted_noise = self.forward(x, torch.Tensor([t]))
            
            # Compute reverse diffusion step parameters
            mult_prefix = 1 / math.sqrt(self.alphas[t])
            predicted_noise_mult = (self.betas[t] / math.sqrt(1 - self.alphas_cumsum[t]))
            
            # Reverse diffusion formula to get x_{t-1} from x_t and predicted noise
            x_prev_t = mult_prefix * (x - (predicted_noise_mult) * predicted_noise + z * math.sqrt(self.betas[t]))
            
        return x_prev_t
    
    def configure_optimizers(self):
        # Configure the Adam optimizer with learning rate 2e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
    
    def forward_noise(self, x_0, t):
        # Utility function to directly apply forward diffusion process to x_0
        # Adds noise to clean images according to timestep t
        
        print(f"Input shape {x_0.shape}")
        sqrt_alpha_cum_batched = self.sqrt_alpha_cum[t]
        eps = torch.randn(x_0.shape)  # Random noise
        # Apply forward diffusion formula
        output = x_0 * (sqrt_alpha_cum_batched) + math.sqrt((1 - self.alphas_cumsum[t])) * eps
        return output

    





