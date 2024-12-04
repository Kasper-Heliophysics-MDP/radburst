from radburst.utils.dataset import Dataset, MinMaxNormalize, Resize
import radburst.utils.preprocessing as prep
from torchvision import transforms
import matplotlib.pyplot as plt
from radburst.detection.ml.models.cnn import CNN
from radburst.detection.ml.predict import Predictor
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
# Provide path to dataset and labels file
data_path = '../../data/fit_files'
labels_path = '../../data/labels/filtered-labels-20240309-20240701.csv'

# Collect functions to preprocess data samples
preprocess_steps = transforms.Compose([
    prep.stan_rows_remove_verts,
    Resize((128,128)),
    MinMaxNormalize()
])

# Create dataset using above settings
dataset = Dataset(
    data_dir= data_path,
    labels= labels_path,
    preprocess= preprocess_steps
)

model = CNN()
path_saved_model = "../../radburst/detection/ml/trained_models/cnn-03-13-2024.pth"

# Load trained state
state_dict = torch.load(path_saved_model, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if 'fc' in name:  # Unfreeze fully connected layers
        param.requires_grad = True
# Split dataset into training and validation sets
train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)

# Create DataLoader objects for training and validation
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Assuming you have defined your model and data loaders
# Replace 'model', 'train_loader', and 'val_loader' with actual implementations
# Directory to save checkpoints
checkpoint_dir = "checkpoints/transferII"
os.makedirs(checkpoint_dir, exist_ok=True)
# Learning rate and optimizer
lr = 0.0005
optimizer = optim.Adam(model.parameters(), lr=lr)

# # Loss function with a positive weight

# # Check device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.3])).to(device)
model.to(device)

# # Number of epochs
num_epochs = 10

# # Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()
    train_loss = 0.0
    
    for inputs, labels in train_loader:
        # Move data to the same device as the model
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, labels.float())  # Ensure labels are float for BCEWithLogitsLoss
        
        # Zero the gradient buffers
        optimizer.zero_grad()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate training loss
        train_loss += loss.item()
    
    # Print training statistics for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}")
    
    # Validation loop (optional)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move data to the same device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            
            # Accumulate validation loss
            val_loss += loss.item()
            
            # Compute accuracy
            predicted = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss / len(train_loader),
        'val_loss': val_loss / len(val_loader),
        'val_accuracy': val_accuracy,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
# Path to the checkpoint file for epoch 3
checkpoint_path = "checkpoints/transferII/model_epoch_8.pth"

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Load the model state dictionary
model.load_state_dict(checkpoint['model_state_dict'])

# Load the optimizer state dictionary (optional if resuming training)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Get additional metadata
epoch = checkpoint['epoch']
train_loss = checkpoint['train_loss']
val_loss = checkpoint['val_loss']
val_accuracy = checkpoint['val_accuracy']

print(f"Checkpoint loaded: Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Evaluate the model
model.eval()
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        # Move data to the same device as the model
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward passval_loader
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        
        # Accumulate validation loss
        val_loss += loss.item()
        
        # Compute accuracy
        predicted = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Calculate and print validation metrics
val_accuracy = 100 * correct / total
print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Evaluate the model and collect predictions and labels
all_labels = []
all_scores = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        # Move data to the same device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        scores = torch.sigmoid(outputs)  # Convert logits to probabilities
        
        # Collect labels and scores
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores.cpu().numpy())

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_scores = np.array(all_scores)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Compute the confusion matrix
threshold = 0.5  # Use a threshold of 0.5 to convert probabilities to binary predictions
predictions = (all_scores > threshold).astype(int)
conf_matrix = confusion_matrix(all_labels, predictions)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Print scores and confusion matrix
print("Confusion Matrix:")
print(conf_matrix)
print(f"AUC: {roc_auc:.2f}")
