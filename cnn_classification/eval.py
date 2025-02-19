from models.cnn import CNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.preprocessing as prep
from utils.dataset import Dataset


def evaluate(args):
    data_path = args['data_path']
    labels_path = args['labels_path']

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
        zip = args['get_data_from_zip']
    )

    model = CNN()
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

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=False)

    # Evaluate the model
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward passval_loader
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            
            # Accumulate validation loss
            test_loss += loss.item()
            
            # Compute accuracy
            predicted = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Calculate and print validation metrics
    test_accuracy = 100 * correct / total
    print(f"test Loss: {test_loss / len(test_loader):.4f}, test Accuracy: {test_accuracy:.2f}%")
    

    # Evaluate the model and collect predictions and labels
    all_labels = []
    all_scores = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
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

if __name__ == "__main__":
    # Load YAML file
    with open("cnn_classification/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    evaluate(config['default'])
