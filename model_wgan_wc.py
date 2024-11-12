import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import save_models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Critic loss function
def critic_loss(real_data_score, fake_data_score):
    return torch.mean(fake_data_score) - torch.mean(real_data_score)

# Generator loss function
def generator_loss(fake_data_score):
    return -torch.mean(fake_data_score)


# Weight clipping function
def clip_weights(model, clip_value):
    for param in model.parameters():
        param.data.clamp_(-clip_value, clip_value)
        
        
# WGAN model training
def train_wgan(critic, generator, real_data_loader, num_epochs, n_critic = 5, lr = 0.00005, clip_value = 0.01, batch_print_frequency = 100):

    # Define optimizers
    optimizer_c = optim.RMSprop(critic.parameters(), lr = lr)  # Critic optimizer
    optimizer_g = optim.RMSprop(generator.parameters(), lr = lr)  # Generator optimizer

    # Move models to device
    critic.to(device)
    generator.to(device)

    # Model training
    for epoch in range(num_epochs):
        for batch_idx, (real_data, _) in enumerate(real_data_loader): 
            real_data = real_data.to(device)  # Move real data to device
            batch_size = real_data.size(0)  # Get size of current batch

            # Reshape real_data to have correct dimensions for critic
            real_data = real_data.view(batch_size, -1)  # Flatten the input

            # Train critic n_critic times
            for _ in range(n_critic):

                # Generate fake data
                noise = torch.randn(batch_size, 100, device = device)  # Create random noise
                fake_data = generator(noise).detach()  # Output fake data from generator

                # Compute critic loss
                optimizer_c.zero_grad()  # Clear previous gradients for critic
                loss_c = critic_loss(critic(real_data), critic(fake_data))  # Calculate critic loss
                loss_c.backward()  # Backpropagate loss
                optimizer_c.step()  # Update critic weights

                # Clip critic weights
                clip_weights(critic, clip_value)

            # Train generator
            noise = torch.randn(batch_size, 100, device = device)  # Create random noise
            fake_data = generator(noise)  # Output fake data from generator

            optimizer_g.zero_grad()  # Clear previous gradients for generator
            loss_g = generator_loss(critic(fake_data))  # Calculate generator loss
            loss_g.backward()  # Backpropagate loss
            optimizer_g.step()  # Update generator weights

        # Save to checkpoints
        if epoch % 10 == 0:
            save_models(generator, critic, 'checkpoints')
