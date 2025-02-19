# Data Download and Processing

**To read in data, download FITfiles from the google drive as a zip file. This will download multiple zip files. Place them all in this data folder**

metadata/                               Folder of csv files for metadata.  

    classification_labels_raw.csv           Someone made manual classifications of sun data. These were png images of spectograms. 
                                            Link to classifications: https://docs.google.com/document/d/1YfonaM4mR5wb6eVOeXuNQUpLTaiJpuKH_D6yrXy9aA4/edit  

    filtered-labels-20240309-20240701.csv   This csv contains rows for every fit file in the google drive from the range of dates 
                                            (which believe is unknown) where classifications were made. __This csv is what you 
                                            should use to read in data.__ 

    full-labels-20240120-20241009.csv       This csv contains rows for every fit file in the google drive as of 
                                            10/18/24 with classifications listed.  

sample_data/                            Just examples to give you a picture of what we're working with.  
  
create_labels.ipynb                     Builds the full labels and filtered labels csv. Just reads files from the drive and 
                                        assigns them their classification as given in classification_labels_raw.csv.  


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

## Future Improvements

1. Fine-tuning hyperparameters for better performance
2. Implementing more advanced data augmentation techniques
3. Exploring ensemble methods or more complex architectures
4. Use data from Peach Mountain to furhter improve generalization

  
 

