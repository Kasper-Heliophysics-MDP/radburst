## ml structure
    
    models/                 # model architectures
    trained_models/         # saved trained models (eg .pth files)
    train.py                # script for training a model
    eval.py                 # script for evaluating perf on val/test set
    predict.py              # make preds on new data (folder or single sample) - this can be function(s) or script

other possible adds:
- utils.py for metrics, predicting
- dataset files/folder for preprocessing and creating dataset