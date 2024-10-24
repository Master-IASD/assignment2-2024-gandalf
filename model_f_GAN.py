import torch
import torch.nn as nn
import torch.nn.functional as F

class fGAN:
    def __init__(self, generator, variational_function, g_optimizer, v_optimizer, divergence):
        self.generator = generator
        self.variational_function = variational_function
        self.g_optimizer = g_optimizer
        self.v_optimizer = v_optimizer
        self.div = divergence()  # f-divergence used (e.g jensen-shannon, etc)
    def train_step(self, real_data, batch_size):
        noise = torch.randn(batch_size, 100)  # Random noise for the generator

        # Generator step
        self.v_optimizer.zero_grad()
        generated_data = self.generator(noise)

        # Variational function step for real data
        real_score = self.variational_function(real_data)
        
        # Variational function step for generated data
        fake_score = self.variational_function(generated_data.detach())

        # f-GAN objective based on the variational lower bound
        v_loss = -(-torch.mean(self.div.f(real_score)) + torch.mean(self.div.f_star(self.div.f(fake_score)))) #-loss because gradient ascent
        v_loss.backward()
        self.v_optimizer.step()

        # Generator update
        self.g_optimizer.zero_grad()
        fake_score_updated = self.variational_function(generated_data)
        g_loss = torch.mean(self.div.f_star(self.div.f(fake_score_updated)))
        g_loss.backward()
        self.g_optimizer.step()

        return v_loss.item(), g_loss.item()
    

# Define f* based on the specific f-divergence (for example, Jensen-Shannon, Kullback-Leibler, etc.)

class jensen_shannon :
    def __init__(self):
        self.name = 'js'
    
    def f(self,t):
        return torch.log(torch.ones(t.shape)*2) -torch.log(1+torch.exp(-t))
    
    def f_star(self,t):
        return -torch.log(2 - torch.exp(t))

class Kullback_Leibler:
    def __init__(self):
        self.name = 'KL'
    
    def f(self,t):
        return t
    
    def f_star(self,t):
        return torch.exp(t-1)

class Pearson:
    def __init__(self):
        self.name = 'Pearson'
    
    def f(self,t):
        return t
    
    def f_star(self,t):
        return 0.25*t**2 + t
