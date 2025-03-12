import matplotlib.pyplot as plt
from tamag.transforms import Denoise, Flip, GaussianBlur, PolarityInversion, Pad, FluxGuided, RandomNoise, ResizeByHalf, Rotate, ByteScaling, BitmapCropping, HistogramEqualization
from astropy.io import fits
import sys
import os
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.utils as utils

plot_save_dir = "C:/Users/15862/OneDrive/Documents/Senior year assignments/helio/data/augmented_images"
def plot_transformation(original, transformed, transform_name, show=False):
    """
    Plots the original and transformed images side by side with a title indicating the transformation name.
    Saves the plot as an image file with the transform name as the filename.

    Args:
        original (numpy.ndarray): The original image array.
        transformed (numpy.ndarray): The transformed image array.
        transform_name (str): The name of the transformation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original.astype(float), aspect='auto')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(transformed.astype(float), aspect='auto')
    axes[1].set_title("Transformed")
    axes[1].axis('off')
    
    plt.suptitle(transform_name)
    plt.tight_layout()
    
    # Save the plot as an image file
    filename = f"{transform_name.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(plot_save_dir, filename), bbox_inches='tight')
    if show:
        plt.show()
    
    print(f"Plot saved as {filename}")

if(1):
    # Clear out directory contents to start
    files = glob.glob(os.path.join(plot_save_dir, "*"))  # Get all files
    for file in files:
        os.remove(file) 

sample_data = "C:/Users/15862/OneDrive/Documents/Senior year assignments/helio/data/sample_data/Beelink1_20240418_130000_59.fit"
fits_full_data = fits.open(sample_data)
spectrogram_arr = utils.load_fits_file(fits_full_data)

denoise = Denoise()
flip_h = Flip(direction='horizontal')
flip_v = Flip(direction='vertical')
gausblur_100 = GaussianBlur(sigma=100)
gausblur_10 = GaussianBlur(sigma=10)
invert = PolarityInversion()
pad = Pad(infer_output_size=True)
flux_10 = FluxGuided(stride=10)
flux_1 = FluxGuided(stride=1)
noise_500 = RandomNoise(gauss=500)
noise_10 = RandomNoise(gause=10)
shrink = ResizeByHalf()
rotate_45 = Rotate(angle=45)
rotate_90 = Rotate(angle=90)
byte = ByteScaling()
bit = BitmapCropping()

plot_transformation(spectrogram_arr, denoise.transform(spectrogram_arr), "Denoise")
plot_transformation(spectrogram_arr, flip_h.transform(spectrogram_arr), "Flipped horizontally")
plot_transformation(spectrogram_arr, flip_v.transform(spectrogram_arr), "Flipped vertically")
plot_transformation(spectrogram_arr, gausblur_100.transform(spectrogram_arr), "Blur (sigma=100)")
plot_transformation(spectrogram_arr, gausblur_10.transform(spectrogram_arr), "Blur (sigma=10)")
plot_transformation(spectrogram_arr, invert.transform(spectrogram_arr.astype(int)), "Invert Polarity")
plot_transformation(spectrogram_arr, pad.transform(spectrogram_arr), "Pad")
plot_transformation(spectrogram_arr, flux_10.transform(spectrogram_arr), "Flux Guiding (stride=10)")
plot_transformation(spectrogram_arr, flux_1.transform(spectrogram_arr), "Flux Guiding (stride=1)")
plot_transformation(spectrogram_arr, noise_500.transform(spectrogram_arr), "Add Noise (mean=500)")
plot_transformation(spectrogram_arr, noise_10.transform(spectrogram_arr), "Add Noise (mean=10)")
plot_transformation(spectrogram_arr, shrink.transform(spectrogram_arr), "Resize By Half")
plot_transformation(spectrogram_arr, rotate_45.transform(spectrogram_arr), "Rotate (angle=45)")
plot_transformation(spectrogram_arr, rotate_90.transform(spectrogram_arr), "Rotate (angle=90)")
plot_transformation(spectrogram_arr, byte.transform(spectrogram_arr), "Byte Scaling")
#plot_transformation(spectrogram_arr, bit.transform(spectrogram_arr), "Bitmap Cropping") #TO DO: extract bitmap from morphops preprocessing