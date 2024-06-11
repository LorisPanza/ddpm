import torch
import matplotlib.pyplot as plt
import random
import numpy as np

from unet import Unet
from ddpm import DDPM
from dataset import Dataset_diffusion_models

def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.permute(0,2,3,1) 
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()


# Shows the first batch of images
def show_first_batch(loader):
    for batch in loader:
        print(batch[0].shape)
        show_images(batch[0], "Images in the first batch")
        break

def show_forward_process(steps,model,image):
    for i in range(steps):
        print(f"Forward process timestamp {i}")
        output=model(image,i)
        plt.imshow(output[0].numpy())
        plt.show()


def training_scheduler(train_data_loader,val_data_loader, epochs=1, timesteps=100):

    ddpm_model = DDPM(timesteps=timesteps,imagechannels=1)
    optimizer = ddpm_model.configure_optimizers()
    training_loss = []
    print(optimizer)

    for epoch in range(epochs):

        for data,_ in train_data_loader:

            optimizer.zero_grad()

            loss = ddpm_model.training_process(data)

            loss.backward()

            optimizer.step()

            print(f"Loss -> {loss.item()}")

            training_loss.append(loss)

    
    for data,_ in val_data_loader:
        
        
        plt.imshow(data.permute(0,2,3,1).squeeze(0).numpy())
        plt.show()
        
        t = torch.randint(0,timesteps,size=(1,))
        noised_image = ddpm_model.forward_noise(data,t)

        plt.imshow(noised_image.permute(0,2,3,1).squeeze(0).numpy())
        plt.show()

        for t_actual in reversed(range(t+1)):
            print((t_actual))
            noised_image = ddpm_model.sampling(noised_image,t_actual)

            plt.imshow(noised_image.permute(0,2,3,1).squeeze(0).numpy())
            plt.show()
        
 


steps_diffusion_process = 200

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Model Parameters
fashion = True
batch_size=16

# DDPM model initialization
ddpm_model = DDPM(steps_diffusion_process)

# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
dataset = Dataset_diffusion_models(batch_size=32)
train_loader, val_loader = dataset.data_loader()

verbose_diffusion = False

if(verbose_diffusion):
    # Showing first batch
    show_first_batch(train_loader)   


training_scheduler(train_loader,val_loader)

