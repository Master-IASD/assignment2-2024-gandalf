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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

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
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim))#.cuda()
    D = torch.nn.DataParallel(Discriminator(d_input_dim = mnist_dim))#.cuda()

    #G = Generator(g_output_dim = mnist_dim)
    #D = Discriminator(mnist_dim)

    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 



    # define loss
    criterion = nn.BCELoss() 

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr/2, betas = (0.5,0.999))
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr, betas = (0.5,0.999))

    #Test defining f-gan
    model = fGAN(generator = G,
                 variational_function = D,
                 g_optimizer = G_optimizer,
                 v_optimizer = D_optimizer,
                 divergence = Kullback_Leibler)
    
    print('Start Training :')
    
    n_epoch = args.epochs
    dloss_final = []
    gloss_final = []
    for epoch in trange(1, n_epoch+1, leave=True):
        dloss = []
        gloss = []          
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            #dloss_tmp = D_train(x, G, D, D_optimizer, criterion)
            #gloss_tmp = G_train(x, G, D, G_optimizer, criterion)

            dloss_tmp,gloss_tmp = model.train_step(real_data=x,batch_size=x.shape[0])
            dloss.append(dloss_tmp)
            gloss.append(gloss_tmp)
        if epoch % 10 == 0:
            save_models(model.generator, model.variational_function, 'checkpoints')

        #print(f'dloss = {np.mean(dloss)}')
        #print(f'gloss = {np.mean(gloss)}')
        dloss_final.append(np.mean(dloss))
        gloss_final.append(np.mean(gloss))

    print('Training done')
    import pylab as plt
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(np.sign(dloss_final)*np.log(np.abs(dloss_final)),marker='.',label='discriminator')
    ax[1].plot(np.sign(gloss_final)*np.log(np.abs(gloss_final)),marker='.',label='generator')
    ax[0].legend()
    plt.show()

        
