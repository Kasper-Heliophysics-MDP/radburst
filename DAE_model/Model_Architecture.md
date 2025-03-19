# Model Architecture

The **Deep Denoising Autoencoder (DDA)** is a **convolutional autoencoder (CAE)** designed to **denoise radio images** while preserving **faint extended structures** from paper [Convolutional Deep Denoising Autoencoders for Radio Astronomical Images](https://doi.org/10.48550/arXiv.2110.08618).

## Architecture

### **Encoder**
- **2 convolutional layers** – Extracts spatial features.
- **2 pooling layers** – Downsamples images to reduce complexity.
- **Latent space (dense layer)** – Compressed representation of the input image.

### **Decoder**
- **1 fully connected layer** – Initiates reconstruction.
- **2 convolutional layers** – Rebuilds spatial details.
- **2 upsampling layers** – Restores the original resolution.
- **Output layer** – Produces the final denoised image.

## Key Design Considerations
- A **deeper network** was tested but led to **excessive blurring** of bright features.
- The **final model** was selected to **optimally remove noise while preserving faint radio sources**.

### Pros
- **Effective Feature Extraction:**  
  The convolutional layers capture essential spatial features needed for denoising, which is crucial in radio images.
- **Dimensionality Reduction:**  
  Pooling layers reduce the complexity of the data while preserving key information, making the network more efficient.
- **Compact Representation:**  
  The latent space (dense layer) compresses the input, helping the network focus on the most relevant features.
- **Balanced Reconstruction:**  
  Upsampling layers work to restore the original image resolution, aiding in the recovery of faint, extended structures.
- **Optimized Depth:**  
  A shallower network was chosen to avoid the excessive blurring seen in deeper architectures, ensuring that bright features remain distinct.
- **Task-Specific Design:**  
  The architecture is tailored to the unique challenges of radio astronomical imaging, balancing noise removal with the preservation of low S/N features.

### Cons
- **Loss of Fine Details:**  
  The pooling and convolution operations may smooth out sharp features, potentially leading to some blurring of important details.
- **Limited Depth:**  
  While preventing blurring, the relatively shallow architecture might not capture very complex noise patterns as effectively as deeper networks.
- **Hyperparameter Sensitivity:**  
  The performance is highly dependent on hyperparameter choices (e.g., kernel sizes, number of neurons in the latent space, learning rate), which may require extensive tuning.
- **Tiling Artefacts:**  
  Dividing images into tiles for processing can introduce boundary artefacts if not managed properly, potentially affecting the overall reconstruction quality.
