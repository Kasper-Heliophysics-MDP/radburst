import torch 
import numpy as np
from tqdm import tqdm

class Predictor:
    """Class to make predictions for a model
    
    Attributes:
        model (torch.nn.Module): The model used for making predictions.
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, data_loader):
        """Make predictions on data provided by data_loader.
        
        Args:
            data_loader (DataLoader): PyTorch DataLoader containing data to make predictions for in batches.
            
        Returns:
            tuple: Two lsts containing predictions (probabilities) and true labels.
        """
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                spects = batch[0]
                labels = batch[1]

                outputs = self.model(spects)
                preds = torch.sigmoid(outputs)

                predictions.extend(preds.numpy())
                true_labels.extend(labels.numpy())

        return np.array(predictions), np.array(true_labels)

                

