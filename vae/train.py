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
print(f'{DEVICE = }')
INPUT_DIM = 32000     # 80 * 400 for spectrograms
BATCH_SIZE = 32
LR = 1e-4

# Dataset Loading
train_dataset = SpectrogramDataset(data_dir, train=True)
test_dataset = SpectrogramDataset(data_dir, train=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


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




    for i in range(num_examples):
        epsilon = torch.randn_like(sigma).to(DEVICE)
        z = mu + sigma * epsilon

def main():
    # Load weights if they exist
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    
    # Train new model if no weights exists
    else:
        train()
        torch.save(model.state_dict(), MODEL_PATH)

    # Compute loss on test data




if __name__ == '__main__':
    main()