from torchvision import transforms as transforms
from models.cnn import CNN
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.preprocessing as prep
from utils.dataset import Dataset

def validate_one_epoch(model, val_loader, criterion, device='cpu'):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move data to the same device
            inputs, labels = batch['spectrogram'].to(device), batch['label'].to(device)
            
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
    return val_loss/len(val_loader), val_accuracy

def train_one_epoch(model, train_loader, criterion, optimizer, device='cpu'):    
    model.train()
    train_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to the same device as the model
        inputs, labels = batch['spectrogram'].to(device), batch['label'].to(device)
        
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

    return train_loss/len(train_loader)
    
def save_checkpoint(checkpoint_dir, epoch, model, optimizer, train_loss, val_loss, val_accuracy):
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def train(args):
    data_path = args['data_path']
    labels_path = args['labels_path']

    # Collect functions to preprocess data samples
    preprocess_steps = transforms.Compose([
        prep.stan_rows_remove_verts,
        #np.resize(new_shape=(190, 3000)),
        #scaler.fit_transform()
        ])

    # Create dataset using above settings
    dataset = Dataset(
        data_dir = data_path,
        labels = labels_path,
        preprocess = preprocess_steps,
        zip = args['get_data_from_zip'],
        resize = (128, 128),
        scaler = MinMaxScaler()
    )

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args['pos_weight']])).to(device) # Loss function, weight positive samples higher 
    model.to(device) # Send model to gpu if used

    # Load trained state
    path_saved_model = args['load_from_checkpoint']
    if path_saved_model:
        state_dict = torch.load(path_saved_model, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'fc' in name:  # Unfreeze fully connected layers
                param.requires_grad = True
        print(f"Checkpoint loaded: Epoch {state_dict['epoch']}, Train Loss: {state_dict['train_loss']:.4f}, Validation Loss: {state_dict['val_loss']:.4f}, Validation Accuracy: {state_dict['val_accuracy']:.2f}%")


    # Split dataset into training and validation sets
    train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=args['pct_test'], random_state=42)

    # Create DataLoader objects for training and validation
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    n_epochs = args['max_epoch']
    n_valid = args['validate_every_n_steps']
    n_ckpt = args['checkpoint_every_n_steps']
    train_loss = 0
    val_loss = 0
    val_accuracy = 0
    ckpt_path = args['checkpoint_directory']

    print("Begin training...\n")
    for epoch in range(1, n_epochs):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Print training statistics for the epoch
        print(f"Epoch {epoch}/{n_epochs}, Train Loss: {train_loss:.4f}")

        if epoch % n_valid == 0:
            val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)

        if epoch & n_ckpt == 0:
            save_checkpoint(ckpt_path, epoch, model, optimizer, train_loss, val_loss, val_accuracy)


if __name__ == "__main__":
    # Load YAML file
    with open("cnn_classification/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    train(config['default'])




    
   
    
    
