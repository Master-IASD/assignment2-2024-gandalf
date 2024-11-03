import torch

class fGAN:
    """
    Definine the class f-GAN.
    
    Attributes 
    ------------------------------------------------------------------
        generator : Generator model
        discriminator : Discriminator model
        g_optimizer : Optimizer for the generator
        d_optimizer : Optimizer for the discriminator
        div : f-divergence of the f-GAN model

    Methods 
    ------------------------------------------------------------------
        variational_function : Apply the activation function (gf in the paper) on the discriminator output
        discriminator_loss : Compute the loss value of the discriminator
        generator_loss : Compute the loss value of the generator
        train_step : Compute the backward path and one optimizer step for generator and discriminator

    """
    
    def initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                
                
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, divergence):
        """    
        Construct the f-GAN model with predefined generator and discriminator architectures.

        Parameters
        ------------------------------------------------------------------
            generator : Generator model from model.py
            discriminator : Discriminator model from model.py
            g_optimizer : Optimizer for the generator
            d_optimizer : Optimizer for the discriminator
            divergence : f-divergence used in the format defined in f_divergences.py
        """

        self.generator = generator
        self.discriminator = discriminator
        # Apply this to both generator and discriminator after their instantiation.
        self.initialize_weights(self.generator)
        self.initialize_weights(self.discriminator)
        self.g_optimizer = g_optimizer
        self.v_optimizer = d_optimizer
        self.div = divergence() 

    def variational_function(self,x):
        """Compute the variationnal function based on the discriminator output. We add to remove the last
        activation in the discriminator (see model.py) because it depends on the f-divergence
        
        Parameter 
        ------------------------------------------------------------------
            x : Torch tensor, sample data 
        """
        return self.div.activation(self.discriminator(x))

    def discriminator_loss(self,real_data, batch_size):
        """Compute the loss for the discriminator defined as E[Tw(x)] - E[f*(Tw(x'))]
        where x are real sample, x' samples produced by the generator Tw is the variational_function
        and E is the expectation.
        
        Parameter 
        ------------------------------------------------------------------
            real_data : Torch tensor, sample of real data
            batch_size : int, self explanatory

        Returns 
        ------------------------------------------------------------------
            loss_real-loss_fake : float, loss value E[Tw(x)] - E[f*(Tw(x'))] (see above)
            real_output : float, discriminator response to real sample Tw(x)
            fake_output : float, discriminator response to fake sample Tw(x')
        """
        z = torch.randn(batch_size, 100)
        fake_data = self.generator(z).detach()

        real_output = self.variational_function(real_data)
        fake_output = self.variational_function(fake_data)

        loss_real = torch.mean(real_output)
        loss_fake = torch.mean(self.div.f_star(fake_output))

        return loss_real - loss_fake, real_output, fake_output
    
    def generator_loss(self,batch_size):
        """Compute the loss for the generator defined as -E[f*(Tw(x'))] (only this term in the objective
        function depends on the generator parameters) where x' samples produced by the generator Tw is the variational_function
        and E is the expectation.
        
        Parameter 
        ------------------------------------------------------------------
            batch_size : int, self explanatory

        Returns 
        ------------------------------------------------------------------
            -torch.mean(self.div.f_star(fake_output)) : float, value of the generator loss -E[f*(Tw(x'))]
        """
        z = torch.randn(batch_size, 100)
        fake_data = self.generator(z)
        fake_output = self.variational_function(fake_data)

        return -torch.mean(self.div.f_star(fake_output))

    def compute_accuracy(self,real_output,fake_output):
        """
        Compute accuracy for the discriminator.

        Parameters
        ------------------------------------------------------------------
            real_output : float, discriminator response to real sample Tw(x) (see discriminator loss)
            fake_output : float, discriminator response to fake sample Tw(x') (see discrimnator loss)
        
        Returns
         ------------------------------------------------------------------
            discriminator accuracy : float in [0,1] coresponding to % of samples correctly
                                    classified as real or generated
        """
        real_correct = (real_output >= self.div.threshold).float().mean().item()
        fake_correct = (fake_output < self.div.threshold).float().mean().item()
        discriminator_accuracy = 0.5 * (real_correct + fake_correct)

        return discriminator_accuracy

    def train_step(self, real_data, batch_size):
        """
        Compute One backward path and optimizer step for generator and discriminator.

        Parameters
        ------------------------------------------------------------------
            real_data : Torch tensor, bathc of real samples x
            batch_size : int, self explanatory
        
        Returns
         ------------------------------------------------------------------
            v_loss : float, loss of the discriminator
            g_loss : float, loss of the generator
            accuracy : float, accuracy of the discriminator
        """

        # Generator training
        self.g_optimizer.zero_grad()
        g_loss = self.generator_loss(batch_size)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1) # Gradient clipping to prevent exploding or vanishing gradients
        self.g_optimizer.step()
         
        # Discriminator training 
        self.v_optimizer.zero_grad()
        v_loss, real_output, fake_output = self.discriminator_loss(real_data, batch_size)
        accuracy = self.compute_accuracy(real_output,fake_output) # Compute accuracy
        (-v_loss).backward() # gradient ascent
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)  # Gradient clipping to prevent exploding or vanishing gradients
        self.v_optimizer.step()
           
        return v_loss.item(), g_loss.item(), accuracy




