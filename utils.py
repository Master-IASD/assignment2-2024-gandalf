import torch
import os

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # Train discriminator on real
    x_real, y_real = x.to(device), torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # Train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1, device=device)

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # Gradient backpropagation & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return D_loss.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # Gradient backpropagation & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.item()


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))


def load_model(G, folder, name='G.pth'):
    ckpt = torch.load(os.path.join(folder, name), map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
