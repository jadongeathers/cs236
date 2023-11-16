import torch
from torch import nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img_to_hidden = nn.Linear(input_dim, h_dim)
        self.hidden_to_mu = nn.Linear(h_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_to_hidden = nn.Linear(z_dim, h_dim)
        self.hidden_to_img = nn.Linear(h_dim, input_dim)

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = torch.sigmoid


    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_to_hidden(x))
        mu, sigma = self.hidden_to_mu(h), self.hidden_to_sigma(h)    # no relu so that mu/sigma can be negative
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_to_hidden(z))
        img_reconstructed = self.sigmoid(self.hidden_to_img(h))     # sigmoid to squish pixel values between 0 and 1
        return img_reconstructed

    def forward(self, x):
        '''
        input image (x) --> encoder --> mean, std --> sample w/ reparametrization trick --> decoder --> output image
        '''
        # encode
        mu, sigma = self.encode(x)

        # sample z w/ reparametrization trick: equivalent to randomly sampling z ~ N(mu, sigma), but allows for backprop
        epsilon = torch.randn_like(sigma)      # tensor filled with random numbers from N(mu=0, sigma=1)
        # epsilon = torch.randn(sigma.shape)
        z = mu + sigma * epsilon

        # decode
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, sigma   # x_reconstructed used for reconstruction loss, mu/sigma used for KL divergence


if __name__ == '__main__':
    '''
    Testing for correct shapes after encoding and decoding images
    '''
    x = torch.randn(4, 28*28)     # batch of 4 images (MNIST is 28x28 = 784)
    vae = VariationalAutoencoder(input_dim=28*28)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)