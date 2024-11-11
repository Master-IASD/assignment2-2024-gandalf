import torch
import numpy as np
import csv
from tqdm import trange
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_wgan_gp import WGAN_GP  # Import WGAN-GP specific classes
from model import Generator, Discriminator
from utils import save_models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
mnist_dim = 784  # 28x28 images flattened

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize models
G = Generator(g_output_dim=mnist_dim).to(device)
D = Discriminator(d_input_dim=mnist_dim).to(device)

# Optimizers
g_optimizer = optim.RMSprop(G.parameters(), lr=5e-5)
d_optimizer = optim.RMSprop(D.parameters(), lr=25e-5)

# Initialize the WGAN-GP model
model = WGAN_GP(generator=G, critic=D, g_optimizer=g_optimizer, c_optimizer=d_optimizer, threshold=0.0, device=device)

# Training parameters
n_epoch = 150  # Change as needed

<<<<<<< HEAD
# Training loop
with open('losses.csv', mode='w') as file:
    writer = csv.writer(file, delimiter=',', lineterminator='\n')
    for epoch in trange(1, n_epoch + 1, leave=True):
        dloss = []
        gloss = []
        acc = []
=======
    print('Model Loading...')
    mnist_dim = 784
    G = Generator(g_output_dim = mnist_dim) #.cuda()
    #G = load_model(G, folder = 'checkpoints',name='G.pth')
    G = torch.nn.DataParallel(G) #.cuda()
    D = Discriminator(d_input_dim = mnist_dim) #.cuda()
    #D = load_model(D,folder = 'checkpoints',name = 'D.pth')
    D = torch.nn.DataParallel(D) #.cuda()
>>>>>>> 3d360d24aeb39d48e5078f27088a8557fecc15d2

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)  # Move data to device
            dloss_tmp, gloss_tmp, acc_tmp = model.train_step(real_data=x, batch_size=x.shape[0])
            dloss.append(dloss_tmp)
            gloss.append(gloss_tmp)
            acc.append(acc_tmp)

        current_dloss = np.mean(dloss)
        current_gloss = np.mean(gloss)
        current_acc = np.mean(acc)

        # Save the model every 2 epochs
        if epoch % 2 == 0:
            save_models(G, D, folder='checkpoints')
        # Logging
        print(f'Epoch {epoch} - Critic loss: {current_dloss}, Generator loss: {current_gloss}, Critic accuracy: {current_acc}')
        writer.writerow([current_dloss, current_gloss, current_acc])

print('Training completed.')