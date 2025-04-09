import tensorflow as tf
import numpy as np
from models.DAE_tf_vert import build_denoising_autoencoder
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import zipfile
from astropy.io import fits
import skimage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.utils as utils
import utils.preprocessing as prep

def plot_sample(title, input, output, pdf):
    """
    Plot input, callisto data, and model for a single datapoint.
    
    Args:
        title (str): file path will be the plot title
        input (np.array): Input data.
        output (np.array): model output.
        pdf (PdfPages): PDF object to save the plots.
    """
    y_range = (85, 5)  # Frequency range from 5 MHz to 85 MHz
    x_range = (0, 15 * 60)  # Time range: 15 minutes
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"{title}")

    plots = [("Input LWA Data", input),  
             ("DAE Output", output)]
    
    for ax, (plot_title, data) in zip(axes, plots):
        data = tf.squeeze(data)
        # Plot 2D data as images (time vs frequency)
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        ax.set_title(plot_title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [MHz]")

        # Adjust frequency axis (assuming data is time-frequency representation)
        ax.set_yticks(np.linspace(0, data.shape[0]-1, 6))  # Adjust for 6 ticks
        ax.set_yticklabels(np.linspace(y_range[0], y_range[1], 6).astype(int))  # Map y ticks to MHz

        # Adjust time axis (assuming the time dimension is the 2nd axis)
        ax.set_xticks(np.linspace(0, data.shape[1]-1, 6))  # Adjust for 6 ticks
        ax.set_xticklabels(np.linspace(x_range[0], x_range[1], 6).astype(int))  # Map x ticks to seconds

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def load_checkpoint(checkpoint_dir, model):
    """
    Loads a model, from the checkpoint in the specified directory.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        model (tf.keras.Model): The model to load the weights into.

    Returns:
        model (tf.keras.Model): The model with loaded weights.
    """
    # Initialize a checkpoint object to restore the state
    checkpoint = tf.train.Checkpoint(
        model=model,
    )
    
    # Load the checkpoint from the directory
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_path:
        checkpoint.restore(checkpoint_path).expect_partial()  # Restore from checkpoint
        print(f"Checkpoint restored from {checkpoint_path}")
    else:
        print("No checkpoint found in the specified directory.")
        return None, None, None  # No checkpoint to load
    
    return model


def prepare_for_inference(input):
    #resize
    resized_input = skimage.transform.resize(input, (190, 190)).squeeze()
    resized_input = np.expand_dims(resized_input, axis=(0, -1))

    #minmax normalize
    min_val = np.min(resized_input)
    max_val = np.max(resized_input)
    normalized_input = (resized_input - min_val) / (max_val - min_val + 1e-8)
    
    #standardize rows
    model_input = prep.stan_rows_remove_verts(normalized_input)

    return model_input

if __name__ == '__main__':

    #init model
    ckpt_path = "DAE_model\checkpoints\Precross"
    model = build_denoising_autoencoder()
    load_checkpoint(ckpt_path, model)

    #input data path
    file_path = "FITfiles/2024-04-07/Beelink1_20240407_194500_59.fit"

    #output pdf file
    pdf_filename = "DAEinference_20240407_194500.pdf"

    #find it in the zip files
    zips =  ['data/FITfiles-20250205T173408Z-001.zip', 'data/FITfiles-20250205T173408Z-002.zip', 'data/FITfiles-20250205T173408Z-003.zip']
    for zip_path in zips:
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        fit_file_name = file_path
        if fit_file_name in zip_ref.namelist():
            file_found = True
            fit_file = zip_ref.open(fit_file_name)
            fits_full_data = fits.open(fit_file)
            spectrogram_arr = utils.load_fits_file(fits_full_data)
    if not file_found:
        print(f"error file: {file_path} not found")
        exit(1)

    with PdfPages(pdf_filename) as pdf:
        prediction = model(prepare_for_inference(spectrogram_arr), training=False)
        plot_sample(file_path, spectrogram_arr, prediction, pdf)
        print(f"Model inference saved to: {pdf_filename}")



