# CNN Classification of Radio Bursts

## Overview

This part of the SunRise Project implements a Convolutional Neural Network (CNN) for classifying solar radio bursts. The model is designed to categorize spectrograms of radio emissions into different types of solar bursts or non-burst events.

## Dataset

The dataset consists of spectrograms of radio emissions from various solar observatories, stored as parquet files. Each spectrogram is associated with metadata, including the type of burst (if any) and the instrument that recorded it.

### Data Loading and Preprocessing

The dataset is loaded using a custom `SpectrogramDataset` class, which handles:

1. Reading the metadata from a CSV file
2. Loading spectrogram data from parquet files
3. Applying preprocessing transformations

Key preprocessing steps include:

- Cleaning the spectrograms (likely removing noise or artifacts)
- Converting the data to PyTorch tensors

```38:44:ML_CNN_multi.ipynb
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Unique values in 'instruments': ['Australia-ASSA_62']\n",
      "extracted:       datetime_start datetime_end        instruments type  \\\n",
```

## Model Architecture

The CNN model consists of the following layers:

1. Four convolutional layers, each followed by ReLU activation, batch normalization, and max pooling
2. A dropout layer for regularization
3. Two fully connected layers

The model can be configured for either binary classification (burst vs. no burst) or multi-class classification (different types of bursts).

## Training

The training process includes:

1. Splitting the data into training and testing sets
2. Using Adam optimizer with a learning rate scheduler
3. Employing either Binary Cross Entropy Loss (for binary classification) or Cross Entropy Loss (for multi-class classification)
4. Training for multiple epochs with batch processing

## Evaluation

The model's performance is evaluated using:

1. Confusion Matrix: Visualizes the model's predictions against true labels
2. Accuracy: Overall correctness of predictions
3. Balanced Accuracy: Accounts for imbalanced datasets by considering both True Positive Rate (TPR) and True Negative Rate (TNR)

```324:331:ML_CNN_multi.ipynb
      "Train with multi-class:\n",
      "torch.Size([32, 4]) torch.Size([32, 4])\n",
      "output: tensor([[ 0.2891, -1.9227, -0.9357, -1.5601],\n",
      "        [-0.1225, -2.0723,  1.9852, -0.4430],\n",
      "        [ 0.0236, -1.4689,  1.0078, -1.6238],\n",
      "        [-2.7489, -2.2627,  1.2177,  0.2870],\n",
      "        [-2.1536, -1.9636,  1.4792,  0.2309],\n",
      "        [-2.7985, -1.3299,  1.2968,  0.1508],\n",
```

## Prediction

The trained model can be used to classify new spectrograms. The process involves:

1. Loading and preprocessing a spectrogram
2. Passing it through the model
3. Interpreting the output (either as a binary classification or multi-class probabilities)

```324:331:ML_CNN_multi.ipynb
...
      "        [1., 0., 0., 0.],\n",
      "        [0., 0., 1., 0.],\n",
      "        [0., 0., 1., 0.],\n",
      "        [1., 0., 0., 0.]], device='cuda:0')\n",
      "Train with multi-class:\n",
      "torch.Size([32, 4]) torch.Size([32, 4])\n",
      "output: tensor([[-6.4650, -5.3131,  2.8992,  0.2268],\n",
      "        [-0.5521, -1.6096,  2.1493, -1.1897],\n",
      "        [ 1.2860, -3.8864, -1.2942, -1.4123],\n",
      "        [-4.0036, -2.8892,  2.4641, -0.4154],\n",
      "        [-0.7482, -1.4934,  0.7346,  0.2381],\n",
      "        [-0.1086, -1.8776,  1.6344, -1.5614],\n",
      "        [-3.2191, -1.1481,  1.1905, -0.1358],\n",
      "        [ 1.6171, -1.4201, -1.7759, -2.8426],\n",
      "        [ 0.3017, -2.3983, -2.0110, -2.4166],\n",
      "        [ 1.5503, -2.5793,  0.7315, -2.5977],\n",
      "        [ 0.8426, -0.8174, -1.2955, -3.0981],\n",
      "        [-2.5153, -3.0238,  1.8705,  0.4420],\n",
      "        [ 1.6581, -2.5503, -1.8992, -4.1697],\n",
      "        [ 1.7377, -2.7747, -1.2929, -4.3570],\n",
      "        [-4.2970, -4.5624,  3.9087,  1.3032],\n",
      "        [ 1.7690, -2.3103, -0.0792, -3.6096],\n",
      "        [ 1.3122, -2.6246, -1.6032, -4.3892],\n",
      "        [ 1.8960, -1.0768, -1.9723, -3.9424],\n",
      "        [-4.1699, -1.5095,  1.0781, -0.5765],\n",
      "        [ 1.6021, -1.2675, -1.0299, -3.7056],\n",
      "        [ 1.7012, -2.4459,  0.5008, -3.0506],\n",
      "        [-3.7627, -4.8165,  4.4679,  1.5939],\n",
      "        [-0.2193, -0.3604,  1.6978, -0.0872],\n",
      "        [ 1.8931, -1.8431,  0.9564, -2.4031],\n",
      "        [ 1.8224, -3.1231, -1.0229, -2.3615],\n",
      "        [ 1.5178, -3.5136, -0.5355, -3.3904],\n",
      "        [ 0.6809, -2.7545, -1.3713, -0.5176],\n",
      "        [-1.4871, -2.4481,  2.6247, -1.0430],\n",
      "        [ 2.2529, -2.8657, -1.4888, -5.3307],\n",
      "        [-3.3483, -3.4983,  3.0162,  1.4963],\n",
      "        [-3.5461, -1.8962,  1.1066,  0.1117],\n",
      "        [ 1.2504, -2.0981, -1.3959, -4.6223]], device='cuda:0',\n",
      "       grad_fn=<AddmmBackward0>)\n",
      "labels: tensor([[0., 0., 1., 0.],\n",
      "        [0., 0., 1., 0.],\n",
      "        [1., 0., 0., 0.],\n",
      "        [0., 0., 1., 0.],\n",
      "        [1., 0., 0., 0.],\n",
      "        [1., 0., 0., 0.],\n",
      "        [0., 0., 0., 1.],\n",
      "        [1., 0., 0., 0.],\n",
```

## Future Improvements

1. Fine-tuning hyperparameters for better performance
2. Implementing more advanced data augmentation techniques
3. Exploring ensemble methods or more complex architectures
4. Use data from Peach Mountain to furhter improve generalization
