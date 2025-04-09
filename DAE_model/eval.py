import torch
from torchvision import transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from dataset.dataset import Dataset
import yaml
from models.DAE_tf_vert import build_denoising_autoencoder
from train import load_checkpoint
import sys
import os
import skimage
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


def evaluate_and_plot_samples(model, val_loader, n_samples=0, pdf_filename="output.pdf"):
    """
    Evaluate a TensorFlow model, take a random sample of datapoints, and plot their inputs, predictions, and actual outputs.

    Args:
        model (tf.keras.Model): Trained TensorFlow model to evaluate.
        val_loader (torch.utils.data.DataLoader): Validation DataLoader in PyTorch.
        n_samples (int): Number of random samples to plot. If 0, no samples are plotted.
        pdf_filename (str): Name of the output PDF file.
    """
    # Initialize PDF for saving plots
    with PdfPages(pdf_filename) as pdf:
        all_plot_titles = []
        all_inputs = []
        all_callistos = []
        all_predictions = []
        all_labels = []
        total_mse = 0.0
        total_psnr = 0.0
        num_batches = 0

        # Evaluate data in batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                titles, inputs, callistos, labels = batch['path'], batch['peach_mountain_spectrogram'], batch['callisto_spectrogram'], batch['label']  # Adjust depending on your data
                inputs, callistos = inputs.cpu().numpy(), callistos.cpu().numpy()
                

                # Convert PyTorch tensors to NumPy arrays for TensorFlow
                inputs = np.array(inputs)
                callistos = np.array(callistos)

                # Get predictions from the TensorFlow model
                predictions = model(inputs, training=False)

                # Compute Mean Squared Error (mse) 
                mse_batch = np.mean((callistos - predictions) ** 2)
                total_mse += mse_batch

                # Compute Peak Signal to Noise Ratio (psnr) for this batch
                psnr_batch = 10.0 * tf.math.log((2^32 - 1)**2 / mse_batch) / tf.math.log(10.0)
                total_psnr += psnr_batch

                # Collect data for later plotting
                for i, idx in enumerate(titles):
                    all_plot_titles.append(titles[i])

                for i, idx in enumerate(labels):
                    all_labels.append(labels[i])
                    
                all_inputs.append(inputs)
                all_callistos.append(callistos)
                all_predictions.append(predictions)
                all_labels.append(labels)

                num_batches += 1

        # Compute average MSE and PSNR
        avg_mse = total_mse / num_batches
        avg_psnr = total_psnr / num_batches

        # Concatenate all data
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_callistos = np.concatenate(all_callistos, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # Sample N random indices
        total_samples = all_inputs.shape[0]
        if n_samples > 0:
            sample_indices = np.random.choice(total_samples, n_samples, replace=False)
        else:
            sample_indices = []

        # Plot and save sampled datapoints
        for i, idx in enumerate(sample_indices):
            plot_sample(all_plot_titles[idx], all_inputs[idx], all_callistos[idx], all_predictions[idx], all_labels[idx], pdf)

        print(f"Test Mean Squared Error: {avg_mse:.4f}")
        print(f"Test Peak Signal-to-Noise Ratio: {avg_psnr:.2f} dB")
        print(f"Plots saved to {pdf_filename}")

def plot_unsized(model, dataset, n_samples=0, pdf_filename="output_unsized.pdf", resize_arg=(128, 128)):
    random_indices = np.random.choice(len(dataset), n_samples, replace=False)
    random_sample = [dataset[i] for i in random_indices]

    with PdfPages(pdf_filename) as pdf:
        for datum in random_sample:
            title, label, input, callisto = datum['path'], datum['label'], datum['peach_mountain_spectrogram'], datum['callisto_spectrogram']
            
            resized_input = skimage.transform.resize(input.numpy(), resize_arg).squeeze()
            resized_input = np.expand_dims(resized_input, axis=(0, -1))
            min_val = np.min(resized_input)
            max_val = np.max(resized_input)
            normalized_input = (resized_input - min_val) / (max_val - min_val + 1e-8)
            model_input = prep.stan_rows_remove_verts(normalized_input)
            prediction = model(model_input, training=False)
            plot_sample(title, input, callisto, prediction, label, pdf)

    print(f"Plots saved to {pdf_filename}")

if __name__ == "__main__":
    # Argument parser 
    with open("DAE_model/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    args = config['default']

    # Load the trained model
    device = '/CPU:0'
    model = build_denoising_autoencoder(device=device)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args['lr'])
    model, optimizer, _ = load_checkpoint(args['checkpoint_directory'], model, optimizer)

    # Create evaluation data 
    resize_arg = None
    if args['resize_t'] and args['resize_f']:
        resize_arg = (args['resize_t'], args['resize_f'])

    

    # Plot the data keeping the input and callisto data the original size in the plots
    unsized_dataset = Dataset(
        data_dir = args['data_path'],
        labels = args['labels_path'],
        zip = args['get_data_from_zip'],
        cache_folder = args['cache_path'],
        resize = None,
        preprocess = False,
        scaler = False,
        verbose = args['verbose']
    )
    
    plot_unsized(model, unsized_dataset.testset(), n_samples=args['eval_num_samples'], pdf_filename=args['unsized_eval_output_pdf_path'], resize_arg=resize_arg)


    # Normal model evaluation
    dataset = Dataset(
        data_dir = args['data_path'],
        labels = args['labels_path'],
        zip = args['get_data_from_zip'],
        cache_folder = args['cache_path'],
        resize = resize_arg,
        preprocess = args['preprocess'],
        scaler = args['use_scaler'],
        verbose = args['verbose']
    )

    eval_dataset = dataset.testset()
    test_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args['batch_size'], shuffle=True)

    # Evaluate and plot
    print("Begin evaluation...\n")
    evaluate_and_plot_samples(model, test_loader, n_samples=args['eval_num_samples'], pdf_filename=args['eval_output_pdf_path'])

    