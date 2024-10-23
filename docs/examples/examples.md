# RadBurst Usage Examples

This section provides practical examples of how to use the RadBurst package for solar radio burst detection and classification.

## 1. Loading and Preprocessing Data

Note: This is a placeholder example of the actual implementation of detection and classification.

```python
from radburst.utils.utils import load_fits_file
from radburst.utils.preprocessing import standardize_rows

# Load the FITS file
spectrogram = load_fits_file('path/to/your/fits_file.fits')

# Preprocess the data
preprocessed_spectrogram = standardize_rows(spectrogram)
```

## 2. Visualizing a Spectrogram

Note: This is a placeholder example of the actual implementation of detection and classification.

```python
from radburst.utils.utils import plot_spectrogram
import matplotlib.pyplot as plt

# Plot the spectrogram
plot_spectrogram(preprocessed_spectrogram)
plt.title('Preprocessed Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(label='Intensity')
plt.show()
```

## 3. Detecting and Classifying Bursts

Note: This is a placeholder example of the actual implementation of detection and classification.

```python
from radburst.detection import detect_bursts
from radburst.classification import classify_bursts

detected_bursts = detect_bursts(preprocessed_spectrogram)

classified_bursts = classify_bursts(detected_bursts)

print(f"Number of detected bursts: {len(detected_bursts)}")
print("Classification results:")
for burst, classification in classified_bursts:
    print(f"Burst at {burst['time']} classified as {classification}")
```
