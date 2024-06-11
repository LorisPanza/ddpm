# Import of libraries
import numpy as np


import torch
import torch.nn as nn


import math

from unet import Unet


class DDPM(nn.Module):
    def __init__(self, timesteps,imagechannels=3, channels=64,time_embedding_size=16):
        super(DDPM,self).__init__()
        self.timesteps = timesteps
        # paper values
        self.max = 0.02
        self.min = 0.0001
        # from min to max considering timesteps step
        self.betas = np.linspace(self.min,self.max,timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumsum = self.alphas.cumprod()
        self.sqrt_alpha_cum = np.sqrt(self.alphas_cumsum)

        self.unet = Unet(image_channels=imagechannels, channels=channels, time_embedding_size=time_embedding_size,ch_mults=(1,2,2))
        self.loss_mse = nn.MSELoss()
    
    def forward(self,x,t):
        return self.unet(x,t)
    
    def training_process(self,batch):
        # Get a random timestamp
        ts = torch.randint(low=0, high=self.timesteps,size=[batch.shape[0]])

        # Generate noise for each image in the batch
        noise = torch.randn(batch.shape)
        noised_images=[]
        
        for i in range(len(ts)):
            sqrt_alpha_cum_batched = self.sqrt_alpha_cum[ts[i]]
            alpha_cum_batched = self.alphas_cumsum[ts[i]]
            noised_image = batch[i] * (sqrt_alpha_cum_batched) + math.sqrt((1-alpha_cum_batched))*noise[i]
            noised_images.append(noised_image)
        
        noised_images = torch.stack(noised_images,dim=0)

        # estimation of the noise
        predicted_noise = self.forward(noised_images,ts)
        loss_error_estimation = self.loss_mse(noise, predicted_noise)

        return loss_error_estimation

    
    def sampling(self,x,t):
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape)
            else:
                z = 0

            predicted_noise = self.forward(x,torch.Tensor([t]))
            mult_prefix = 1/math.sqrt(self.alphas[t])
            predicted_noise_mult = (self.betas[t] / math.sqrt(1-self.alphas_cumsum[t]))
            x_prev_t =mult_prefix*(x - (predicted_noise_mult)*predicted_noise +  z*math.sqrt(self.betas[t]))

        return x_prev_t

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer

    def forward_noise(self,x_0,t):
        print(f"Input shape {x_0.shape}")
        sqrt_alpha_cum_batched = self.sqrt_alpha_cum[t]
        eps = torch.randn(x_0.shape)
        output = x_0 *(sqrt_alpha_cum_batched) + math.sqrt((1-self.alphas_cumsum[t]))*eps
        return output

    





