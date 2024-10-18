import matplotlib.pyplot as plt
from astropy.io import fits

def load_fits_file(fits_file_path, num_freq_chans_to_remove=10):
    """Load spectrogram from a FITS into a numpy array.

    Args:
        fits_file_path (str): Path to .fits file containing the spectrogram data.
        num_freq_chans_to_remove (int): Optional parameter to remove low frequency channels with bad signal.

    Returns:
        np.ndarray: Array 
    """
    fits_full_data = fits.open(fits_file_path)
    fits_array = fits_full_data[0].data[:-num_freq_chans_to_remove,:]
    return fits_array


def plot_spectrogram(spect):
    """Plot a spectrogram array.

    Args:
        spect (np.ndarray): Array of spectrogram data.

    Returns:
        None
    """
    plt.imshow(spect, aspect='auto')    