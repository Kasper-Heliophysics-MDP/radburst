from torchvision import transforms as transforms
from models.DAE_tf import build_denoising_autoencoder
import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import yaml
from dataset.dataset import Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.preprocessing as prep

def load_checkpoint(checkpoint_dir, model, optimizer):
    """
    Loads a model, optimizer, and epoch from the checkpoint in the specified directory.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        model (tf.keras.Model): The model to load the weights into.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to load the state into.

    Returns:
        model (tf.keras.Model): The model with loaded weights.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer with loaded state.
        epoch (int): The epoch at which the checkpoint was saved.
    """
    # Initialize a checkpoint object to restore the state
    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0, dtype=tf.int32),  # Default epoch value
        model=model,
        optimizer=optimizer
    )
    
    # Load the checkpoint from the directory
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_path:
        checkpoint.restore(checkpoint_path).expect_partial()  # Restore from checkpoint
        print(f"Checkpoint restored from {checkpoint_path}")
    else:
        print("No checkpoint found in the specified directory.")
        return None, None, None  # No checkpoint to load
    
    # Load the metrics file to get additional info if needed
    metrics_path = os.path.join(checkpoint_dir, f"metrics_epoch_{checkpoint.epoch.numpy()}.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = f.readlines()
            print("Metrics for the loaded checkpoint:")
            for line in metrics:
                print(line.strip())
    else:
        print("Metrics file not found.")
    
    return model, optimizer, checkpoint.epoch.numpy()

def save_checkpoint(checkpoint_dir, epoch, model, optimizer, train_loss, val_loss, val_psnr):
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the checkpoint object
    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(epoch, dtype=tf.int32),  
        model=model,                               
        optimizer=optimizer                        
    )

    # Save additional metrics to a separate file if needed
    metrics_path = os.path.join(checkpoint_dir, f"metrics_epoch_{epoch}.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        f.write(f"Validation PSNR: {val_psnr:.2f}dB\n")

    # Save the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}")
    checkpoint.save(checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def validate_one_epoch(model, val_loader, criterion, device='cpu'):
    total_loss = 0.0
    total_psnr = 0.0
    
    for batch_idx, batch in enumerate(val_loader):
        # Extract inputs and labels from the batch
        inputs, callistos = batch['peach_mountain_spectrogram'].to(device), batch['callisto_spectrogram'].to(device)
        
        # Forward pass
        outputs = model(inputs, training=False)  # Disable training-specific operations like dropout
        
        # Compute loss
        loss = criterion(callistos, outputs)
        total_loss += loss.numpy()
        
        # Compute peak signal to noise ratio
        total_psnr += 10.0 * tf.math.log((2^32 - 1)**2 / loss) / tf.math.log(10.0)
    
    # Compute average loss and psnr
    avg_val_loss = total_loss / len(val_loader)
    avg_val_psnr = total_psnr / len(val_loader)

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Average PSNR: {avg_val_psnr:.2f}dB")
    return avg_val_loss, avg_val_psnr

def train_one_epoch(model, train_loader, criterion, optimizer, device='cpu'):    
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to the same device as the model
        inputs, callistos = batch['peach_mountain_spectrogram'].to(device), batch['callisto_spectrogram'].to(device)

        #forward pass
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = criterion(callistos, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        

    return loss/len(train_loader)

def train(args):

    resize_arg = None
    if args['resize_t'] and args['resize_f']:
        resize_arg = (args['resize_t'], args['resize_f'])

    # Create dataset using settings from config file
    dataset = Dataset(
        data_dir = args['data_path'],
        labels = args['labels_path'],
        zip = args['get_data_from_zip'],
        cache_folder = args['cache_path'],
        resize = resize_arg,
    )

    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    model = build_denoising_autoencoder(device=device)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args['lr'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
    criterion = tf.keras.losses.MeanSquaredError()

    # Load trained state
    starting_epoch = 1
    if args['load_from_checkpoint']:
        model, optimizer, starting_epoch = load_checkpoint(args['checkpoint_directory'], model, optimizer)

    # Create DataLoader objects for training and validation
    train_dataset = dataset.trainset()
    val_dataset = dataset.validset()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    max_epoch = starting_epoch + args['max_epoch']
    n_valid = args['validate_every_n_steps']
    n_ckpt = args['checkpoint_every_n_steps']
    train_loss = 0
    val_loss = 0
    val_psnr = 0
    ckpt_path = args['checkpoint_directory']

    print("Begin training...\n")
    for epoch in range(starting_epoch + 1, max_epoch + 1):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Print training statistics for the epoch
        print(f"Epoch {epoch}/{max_epoch}, Train Loss: {train_loss:.4f}")

        if epoch % n_valid == 0:
            val_loss, val_psnr = validate_one_epoch(model, val_loader, criterion, device)

        if epoch & n_ckpt == 0:
            save_checkpoint(ckpt_path, epoch, model, optimizer, train_loss, val_loss, val_psnr)


if __name__ == "__main__":
    # Load YAML file
    with open("DAE_model/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    train(config['default'])




    
   
    
    
