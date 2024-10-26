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


    def discriminator_loss(self,real_data,fake_data):
        real_output = self.div.f(self.variational_function(real_data)) + torch.randn_like(real_data) * 0.05
        fake_output = self.div.f(self.variational_function(fake_data)) + torch.randn_like(fake_data) * 0.05

        loss_real = torch.mean(real_output)
        loss_fake = torch.mean(self.div.f_star(fake_output))

        return loss_fake - loss_real
    
    def generator_loss(self,fake_data):
        fake_output = self.div.f(self.variational_function(fake_data))
        return torch.mean(self.div.f_star(fake_output))

    def train_step(self, real_data, batch_size):
        noise = torch.randn(batch_size, 100)*0.5  # Random noise for the generator
        generated_data = self.generator(noise)

        # Train the discriminator two times before updating generator
         
        self.v_optimizer.zero_grad()
        v_loss = self.discriminator_loss(real_data,generated_data.detach())
        (-v_loss).backward() #Gradient ascent
        self.v_optimizer.step()

        # Generator update
        
        
        #fake_score_updated = self.variational_function(generated_data)
        
        
        self.g_optimizer.zero_grad()
        g_loss = self.generator_loss(generated_data)
        g_loss.backward()
        self.g_optimizer.step()

        return v_loss.item(), g_loss.item()
