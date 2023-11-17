import os
import torch
import torchvision.datasets as datasets
from tqdm import tqdm

from torch import nn, optim
from model import VariationalAutoencoder
from dataset import SpectrogramDataset
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Configuration
MODEL_PATH = 'mode.pth'
DEVICE = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
print(f'{DEVICE = }')
INPUT_DIM = 32000     # 80 * 400 for spectrograms
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-4

# Dataset Loading
PROJECT_DIR = '/Users/jadongeathers/Desktop/Stanford University/AY 2023-2024/Autumn 2023/CS 236/Final Project/cs236/'
LJSPEECH_DIR = os.path.join(PROJECT_DIR, 'LJSpeech')
data_dir = os.path.join(LJSPEECH_DIR, 'spectrograms/')
train_dataset = SpectrogramDataset(data_dir, train=True)
test_dataset = SpectrogramDataset(data_dir, train=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = VariationalAutoencoder(INPUT_DIM, h_dim=H_DIM, z_dim=Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss(reduction='sum')
# loss_fn = nn.MSELoss(reduction='sum')

# Tensorboard
LOG_DIR = 'tensorboard'
writer = SummaryWriter(LOG_DIR)




def train():
    for epoch in range(NUM_EPOCHS):
        epoch_reconstruction_loss = 0
        epoch_kl_divergence = 0
        epoch_loss = 0
        
        for x in tqdm(train_loader):
            # forward pass
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)    # keep batch dimension (x.shape[0]), but flatten the images
            x_reconstructed, mu, sigma = model(x)

            # compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_divergence = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # backprop
            loss = reconstruction_loss + kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute epoch losses (for tensorboard plots)
            epoch_reconstruction_loss += reconstruction_loss
            epoch_kl_divergence += kl_divergence
            epoch_loss += loss

        # tensorboard
        writer.add_scalar('Epoch Reconstruction Loss', epoch_reconstruction_loss, epoch)
        writer.add_scalar('Epoch KL Divergence', epoch_kl_divergence, epoch)
        writer.add_scalar('Epoch Loss', epoch_loss, epoch)
        print(f'epoch loss: {epoch_loss}')


def test():
    test_loss = 0
    for i, x in tqdm(enumerate(test_loader)):
        # forward pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)    # keep batch dimension (x.shape[0]), but flatten the images
        x_reconstructed, mu, sigma = model(x)

        # compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_divergence = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        loss = reconstruction_loss + kl_divergence

        test_loss += loss
    
    print(f'test loss: {test_loss}')



def inference(digit, num_examples=1):
    # 1. Get an image containing the requested digit
    original_image = None
    for x, y in test_dataset:
        if y == digit:
            original_image = x.to(DEVICE)
            break
    
    # 2. Encode the image to get mu and sigma
    with torch.no_grad():
        mu, sigma = model.encode(original_image.view(1, 784))
    
    # 3. Sample z ~ N(mu, sigma) with reparam trick, decode z, and save image
    for i in range(num_examples):
        epsilon = torch.randn_like(sigma).to(DEVICE)
        z = mu + sigma * epsilon
        reconstructed_image = model.decode(z)
        reconstructed_image = reconstructed_image.view(-1, 1, 28, 28)
        save_image(reconstructed_image, f"results/reconstructed_{digit}_ex{i}.png")       # fun fact: you can save a batch of images (e.g. shape (32, 1, 28, 28)) -- it will save as a single collage image
        

def main():
    # Load weights if they exist
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    
    # Train new model if no weights exists
    else:
        train()
        torch.save(model.state_dict(), MODEL_PATH)

    # Compute loss on test data
    test()

    # Run inference on digits 0-9
    # for digit in range(10):
    #     inference(digit)
    # TODO: Inference on audio

if __name__ == '__main__':
    main()