# checkpoints.py
import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, rewards, epsilon, checkpoint_dir='./checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f'dqn_checkpoint_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'rewards': rewards,
        'epsilon': epsilon,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    '''
    Load checkpoint to resume training.

    inputs:
        model: model to load the state dict into.
        optimizer: optimizer to load the state dict into.
        checkpoint_path: path to the saved checkpoint file.

    Returns:
        model, optimizer, epoch, loss
    '''
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path}")
    
    return model, optimizer, epoch, loss
