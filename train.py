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

import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for ADAM")
    parser.add_argument("--wd", type=float, default=1e-4, 
                        help="Value for weight decay of the discriminator")                
    parser.add_argument("--pretrained", type=bool, default=False, 
                        help="Use pretrained JS model") 
    
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
    G = Generator(g_output_dim = mnist_dim)#.cuda()    
    D = Discriminator(d_input_dim = mnist_dim)#.cuda()

    if args.pretrained :
        D = load_model(D,folder = 'checkpoints',name = 'DJS.pth')
        G = load_model(G, folder = 'checkpoints',name='GJS.pth')

    G = torch.nn.DataParallel(G)#.cuda()
    D = torch.nn.DataParallel(D)#.cuda()

    # model = DataParallel(model).cuda()
    print('Model loaded.')

    lr_G = args.lr
    lr_D = args.lr

    # Define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = lr_G, betas = (0.5,0.999))
    D_optimizer = optim.Adam(D.parameters(), lr = lr_D,betas = (0.5,0.999),weight_decay=args.wd)


    model = fGAN(generator = G,
                 discriminator = D,
                 g_optimizer = G_optimizer,
                 d_optimizer = D_optimizer,
                 divergence = Pearson_chi2)
    
    print('----------------------------------------------')
    print('         Model trained overview ')
    print('----------------------------------------------')
    print(f'Weight decay = {args.wd}')
    print(f'Initial Generator learning rate = {G_optimizer.param_groups[0]['lr']}')
    print(f'Initial discriminator learning rate = {D_optimizer.param_groups[0]['lr']}')
    print(f'f-divergence used : {model.div.name}')
    print(f'Use pretrained JS model : {args.pretrained}')
    print('----------------------------------------------')
    
    print('Start Training :')
    
    n_epoch = args.epochs

    with open('losses.csv',mode='w') as file:
        writer = csv.writer(file,delimiter = ',',lineterminator='\n')
        for epoch in trange(1, n_epoch+1, leave=True):
            dloss = [] # loss values for the discriminator per batch
            gloss = [] # loss values for the generator per batch
            acc = []

            # Warm up when changing f-divergence
            if epoch <= 5 :
                G_optimizer.param_groups[0]['lr'] = lr_G / 100
                D_optimizer.param_groups[0]['lr'] = lr_D / 100

            elif (epoch > 5) and (epoch <= 10):
                G_optimizer.param_groups[0]['lr'] = lr_G / 10
                D_optimizer.param_groups[0]['lr'] = lr_D / 10
            
            elif epoch > 10 :
                G_optimizer.param_groups[0]['lr'] = lr_G
                D_optimizer.param_groups[0]['lr'] = lr_D

            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(-1, mnist_dim)
                dloss_tmp,gloss_tmp,acc_tmp = model.train_step(real_data=x,batch_size=x.shape[0])
                dloss.append(dloss_tmp)
                gloss.append(gloss_tmp)
                acc.append(acc_tmp)

            current_dloss = np.mean(dloss)
            current_gloss = np.mean(gloss)
            current_acc = np.mean(acc)

            if epoch % 5 == 0:
                save_models(model.generator, model.discriminator, 'checkpoints')

            print(f'Discriminator loss : {'{:.1e}'.format(current_dloss)}')
            print(f'Generator loss : {'{:.1e}'.format(current_gloss)}')
            print(f'Accuracy of the discriminator : {'{:.2f}'.format(current_acc)}')
            writer.writerow([current_dloss,current_gloss,current_acc])

    print('Training done')