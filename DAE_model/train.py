from torchvision import transforms as transforms
from models.DAE_tf_vert import build_denoising_autoencoder
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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.preprocessing as prep

def plot_sample(title, input, callisto, output, label, pdf):
    """
    Plot input, callisto data, and model for a single datapoint.
    
    Args:
        title (str): file path will be the plot title
        input (np.array): Input data.
        callisto (np.array): callisto data.
        output (np.array): model output.
        label (int): hand labelled burst type.
        pdf (PdfPages): PDF object to save the plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title}")

    plots = [("Input LWA Data", input), 
             ("eCallisto Data", callisto), 
             ("DAE Output", output)]
    
    for ax, (plot_title, data) in zip(axes, plots):
        data = tf.squeeze(data)
        ax.imshow(data, cmap="viridis", aspect="auto")
        ax.set_title(plot_title)
        ax.axis("off")

        # Add label text box in the top-right corner
        ax.text(0.95, 0.05, f"Label: {label}", 
                transform=ax.transAxes, fontsize=10,
                color="white", backgroundcolor="black",
                ha="right", va="bottom")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

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

def validate_one_epoch(model, val_loader, criterion, device='cpu', pdf_path="output.pdf", N=0):
    total_loss = 0.0
    total_psnr = 0.0
    all_plot_titles = []
    all_labels = []
    all_inputs = []
    all_callistos = []
    all_predictions = []
    
    for batch_idx, batch in enumerate(val_loader):
        # Extract inputs and labels from the batch
        inputs, callistos = batch['peach_mountain_spectrogram'].cpu().numpy(), batch['callisto_spectrogram'].cpu().numpy()
        titles, labels = batch['path'], batch['label']
        # Forward pass
        outputs = model(inputs, training=False)  # Disable training-specific operations like dropout
        
        # Compute loss
        loss = criterion(callistos, outputs)
        total_loss += loss.numpy()
        
        # Compute peak signal to noise ratio
        total_psnr += 10.0 * tf.math.log((2^32 - 1)**2 / loss) / tf.math.log(10.0)

        inputs = np.array(inputs)
        callistos = np.array(callistos)
        
        for i, idx in enumerate(titles):
            all_plot_titles.append(titles[i])

        for i, idx in enumerate(labels):
            all_labels.append(labels[i])
            
        all_inputs.append(inputs)
        all_callistos.append(callistos)
        all_predictions.append(outputs)
        all_labels.append(labels)
    
    # Compute average loss and psnr
    avg_val_loss = total_loss / len(val_loader)
    avg_val_psnr = total_psnr / len(val_loader)

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Average PSNR: {avg_val_psnr:.2f}dB")

    # Sample N random indices
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_callistos = np.concatenate(all_callistos, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    total_samples = all_inputs.shape[0]
    if N > 0:
        sample_indices = np.random.choice(total_samples, N, replace=False)
    else:
        sample_indices = []

    # Plot and save sampled datapoints
    with PdfPages(pdf_path) as pdf:
        for i, idx in enumerate(sample_indices):
            plot_sample(all_plot_titles[idx], all_inputs[idx], all_callistos[idx], all_predictions[idx], all_labels[idx], pdf)

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
        scaler = args['use_scaler'],
        preprocess = args['preprocess'],
        verbose = args['verbose'],
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
            dirname, basename = os.path.split(args['valid_output_pdf_path'])  # Split into directory and filename
            name, ext = os.path.splitext(basename)  # Separate name and extension
            new_filename = f"{name}_epoch_{epoch}{ext}"  # Insert epoch number
            output_path = os.path.join(dirname, new_filename)
            val_loss, val_psnr = validate_one_epoch(model, val_loader, criterion, device, pdf_path=output_path, N=args['valid_print_num_samples'])

        if epoch % n_ckpt == 0:
            save_checkpoint(ckpt_path, epoch, model, optimizer, train_loss, val_loss, val_psnr)


if __name__ == "__main__":
    # Load YAML file
    with open("DAE_model/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    train(config['default'])




    
   
    
    
