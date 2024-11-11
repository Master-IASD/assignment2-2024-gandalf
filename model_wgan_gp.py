import torch
import torch.autograd as autograd

class WGAN_GP:

    def __init__(self, generator, critic, g_optimizer, c_optimizer, threshold=0.0, device='cpu'):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.threshold = threshold  # Set threshold to classify real vs fake image
        self.device = device

    # Generator loss function
    def generator_loss(self, fake_data_score):
        return -torch.mean(fake_data_score)
    
    # Critic loss function
    def critic_loss(self, real_data_score, fake_data_score):
        return torch.mean(fake_data_score) - torch.mean(real_data_score)

    # Gradient penalty function
    def gradient_penalty(self, real_data, fake_data, lambda_gp=10):

        batch_size = real_data.size(0)

        # Set epsilons randomly between 0 and 1, move to device
        epsilon = torch.rand(batch_size, 1, device=self.device)
        epsilon = epsilon.expand_as(real_data)

        # Define interpolated samples and move to device
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        # Calculate critic scores for interpolated samples
        interpolated_scores = self.critic(interpolated)

        # Compute gradients with respect to interpolated samples
        gradients = autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores, device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return penalty
    
    # Accuracy function
    def compute_accuracy(self, real_output, fake_output):

        real_correct = (real_output >= self.threshold).float().mean().item()
        fake_correct = (fake_output < self.threshold).float().mean().item()
        discriminator_accuracy = 0.5 * (real_correct + fake_correct)
        
        return discriminator_accuracy
    
    # WGAN model training
    def train_step(self, real_data, batch_size):

        n_critic = 5
        lambda_gp = 10

        real_data = real_data.to(self.device)
        batch_size = real_data.size(0)  # Get size of current batch

        # Train critic n_critic times
        for _ in range(n_critic):
            noise = torch.randn(batch_size, 100, device=self.device)  # Create random noise on device
            fake_data = self.generator(noise).detach()  # Output fake data from generator
            accuracy = self.compute_accuracy(self.critic(real_data), self.critic(fake_data)) # Compute accuracy
            self.c_optimizer.zero_grad()  # Clear previous gradients for critic
            loss_c = self.critic_loss(self.critic(real_data), self.critic(fake_data))  # Calculate critic loss
            gp = self.gradient_penalty(real_data, fake_data, lambda_gp) # Gradient penalty
            loss_c += gp  # Add gradient penalty to critic loss
            loss_c.backward()  # Backpropagate loss
            self.c_optimizer.step()  # Update critic weights

        # Train generator
        noise = torch.randn(batch_size, 100, device=self.device)  # Create random noise on device
        fake_data = self.generator(noise)  # Output fake data from generator
        self.g_optimizer.zero_grad()  # Clear previous gradients for generator
        loss_g = self.generator_loss(self.critic(fake_data))  # Calculate generator loss
        loss_g.backward()  # Backpropagate loss
        self.g_optimizer.step()  # Update generator weights

        return loss_c.item(), loss_g.item(), accuracy