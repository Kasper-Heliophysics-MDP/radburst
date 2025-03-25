import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from dataset.dataset import Dataset
import yaml
from models.DAE_tf import build_denoising_autoencoder
from train import load_checkpoint

def plot_sample(features, predicted, actual, index, pdf):
    """
    Plot input, predicted output, and actual output for a single datapoint.
    
    Args:
        features (np.array): Input data.
        predicted (np.array): Predicted output.
        actual (np.array): Actual output.
        index (int): Index of the datapoint.
        pdf (PdfPages): PDF object to save the plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Sample {index + 1}")

    # Plot Input Features
    axes[0].imshow(features.squeeze(), cmap="viridis", aspect="auto")
    axes[0].set_title("Input LWA Data")
    axes[0].axis("off")

    # Plot Predicted Output
    axes[1].imshow(predicted, cmap="viridis", aspect="auto")
    axes[1].set_title("eCallisto Data")
    axes[1].axis("off")

    # Plot Actual Output
    axes[2].imshow(actual, cmap="viridis", aspect="auto")
    axes[2].set_title("DAE Output")
    axes[2].axis("off")

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
        all_inputs = []
        all_callistos = []
        all_predictions = []
        total_mse = 0.0
        total_psnr = 0.0
        num_batches = 0

        # Evaluate data in batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs, callistos = batch['peach_mountain_spectrogram'], batch['callisto_spectrogram']  # Adjust depending on your data
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
                all_inputs.append(inputs)
                all_callistos.append(callistos)
                all_predictions.append(predictions)

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
            plot_sample(all_inputs[idx], all_predictions[idx], all_callistos[idx], i, pdf)

        print(f"Test Mean Squared Error: {avg_mse:.4f}")
        print(f"Test Peak Signal-to-Noise Ratio: {avg_psnr:.2f} dB")
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

    dataset = Dataset(
        data_dir = args['data_path'],
        labels = args['labels_path'],
        zip = args['get_data_from_zip'],
        cache_folder = args['cache_path'],
        resize = resize_arg,
    )

    eval_dataset = dataset.testset()
    test_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args['batch_size'], shuffle=True)

    # Evaluate and plot
    evaluate_and_plot_samples(model, test_loader, n_samples=args['eval_num_samples'], pdf_filename=args['eval_output_pdf_path'])