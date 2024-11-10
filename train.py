import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import Generator, Discriminator
from utils import D_train, G_train, save_models

from model_f_GAN import fGAN
from f_divergences import *
from utils import load_model

from model_wgan_gp import WGAN_GP

import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for ADAM")

    args = parser.parse_args()


    os.makedirs('chekpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = Generator(g_output_dim = mnist_dim) #.cuda()
    G = load_model(G, folder = 'checkpoints',name='GJS.pth')
    G = torch.nn.DataParallel(G) #.cuda()
    D = Discriminator(d_input_dim = mnist_dim) #.cuda()
    D = load_model(D,folder = 'checkpoints',name = 'DJS.pth')
    D = torch.nn.DataParallel(D) #.cuda()

    # model = DataParallel(model).cuda()
    print('Model loaded.')

    """
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr, betas = (0.5,0.999))
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr, betas = (0.5,0.999),weight_decay=1e-3)
    """

    g_optimizer = optim.RMSprop(G.parameters(), lr=5e-5)  
    c_optimizer = optim.RMSprop(D.parameters(), lr=25e-5)  

    """
    model = fGAN(generator = G,
                 discriminator = D,
                 g_optimizer = G_optimizer,
                 d_optimizer = D_optimizer,
                 divergence = reverse_KL)
    """
    
    model = WGAN_GP(generator = G,
                 critic = D,
                 g_optimizer = g_optimizer,
                 c_optimizer = c_optimizer,
                 threshold=0.0)
    
    
    # print('Start Training :')
    
    n_epoch = args.epochs

    with open('losses.csv',mode='w') as file:
        writer = csv.writer(file,delimiter = ',',lineterminator='\n')
        for epoch in trange(1, n_epoch+1, leave=True):
            dloss = [] # loss values for the discriminator per batch
            gloss = [] # loss values for the generator per batch
            acc = []      
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(-1, mnist_dim)
                dloss_tmp,gloss_tmp,acc_tmp = model.train_step(real_data=x,batch_size=x.shape[0])
                dloss.append(dloss_tmp)
                gloss.append(gloss_tmp)
                acc.append(acc_tmp)

            current_dloss = np.mean(dloss)
            current_gloss = np.mean(gloss)
            current_acc = np.mean(acc)
            
            if epoch % 2 == 0:
                save_models(model.generator, model.discriminator, 'checkpoints')

            print(f'Discriminator loss : {current_dloss}, Generator loss : {current_gloss}, Discriminator accuracy : {current_acc}')
            writer.writerow([current_dloss,current_gloss,current_acc])

    print('Training done')