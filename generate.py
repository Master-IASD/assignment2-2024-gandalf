import torch
import torchvision
import os
import argparse

from model import Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="The batch size to use for generating samples.")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784
    model = Generator(g_output_dim=mnist_dim).to(device)
    model = load_model(model, name='G.pth', folder='checkpoints').to(device)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    print('Model loaded.')
    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples < 10000:
            z = torch.randn(args.batch_size, 100, device=device)
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples < 10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1