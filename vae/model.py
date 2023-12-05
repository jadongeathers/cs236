import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super().__init__()
        h1_dim = 800
        h2_dim = 400
        self.x_to_h1 = nn.Linear(input_dim, h1_dim)
        self.h1_to_h2 = nn.Linear(h1_dim, h2_dim)
        self.h2_to_z = nn.Linear(h2_dim, z_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.x_to_h1(x))
        out = self.relu(self.h1_to_h2(out))
        out = self.h2_to_z(out)     # no relu so that mu/sigma can be negative
        return out

class Decoder(nn.Module):
    def __init__(self, z_dim, output_dim):
        super().__init__()
        h1_dim = 800
        h2_dim = 400
        self.z_to_h2 = nn.Linear(z_dim, h2_dim)
        self.h2_to_h1 = nn.Linear(h2_dim, h1_dim)
        self.h1_to_x = nn.Linear(h1_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.z_to_h2(x))
        out = self.relu(self.h2_to_h1(out))
        out = self.h1_to_x(out)
        return out

# class Encoder(nn.Module):
#     def __init__(self, input_dim, z_dim):
#         super().__init__()
#         self.x_to_h = nn.Linear(input_dim, 200)
#         self.h_to_z = nn.Linear(200, z_dim)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         out = self.relu(self.x_to_h(x))
#         out = self.h_to_z(out)     # no relu so that mu/sigma can be negative
#         return out

# class Decoder(nn.Module):
#     def __init__(self, z_dim, output_dim):
#         super().__init__()
#         self.z_to_h = nn.Linear(z_dim, 200)
#         self.h_to_x = nn.Linear(200, output_dim)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         out = self.relu(self.z_to_h(x))
#         out = self.h_to_x(out)
#         return out


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=32000, z_dim=800):
        super().__init__()

        # encoder
        self.img_to_mu = Encoder(input_dim, z_dim)
        self.img_to_sigma = Encoder(input_dim, z_dim)

        # decoder
        self.z_to_img = Decoder(z_dim, output_dim=input_dim)

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = torch.sigmoid


    def encode(self, x):
        # q_phi(z|x)
        mu, sigma = self.img_to_mu(x), self.img_to_sigma(x)    # no relu so that mu/sigma can be negative
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        img_reconstructed = self.sigmoid(self.z_to_img(z))     # sigmoid to squish pixel values between 0 and 1
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
    x = torch.randn(4, 32000)     # batch of 4 images (MNIST is 28x28 = 784)
    vae = VariationalAutoencoder()
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)