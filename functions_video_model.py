import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

def train_epoch(model, optimizer, criterion, metrics, dataloader, device):
    model.train()  
    epoch_loss = 0
    epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    for batch in tqdm(dataloader):
        # Move batch to device
        video_features = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()  

        # Forward pass
        outputs = model(video_features)

        # Compute loss 
        loss = criterion(outputs, labels) 

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Get the predictions
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)

        # Compute metrics
        for k in epoch_metrics.keys():
            epoch_metrics[k] += metrics[k](preds.cpu().numpy(), labels.cpu().numpy())

        # Add the loss to the epoch loss  
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    for k in epoch_metrics.keys():
        epoch_metrics[k] /= len(dataloader)

    clear_output() 
    print('train Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))

    return epoch_loss, epoch_metrics


def evaluate(model, criterion, metrics, dataloader, device):
    model.eval() 
    epoch_loss = 0
    epoch_metrics = dict(zip(metrics.keys(), [0]*len(metrics)))
    epoch_preds = []
    epoch_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move batch to device
            video_features = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass
            outputs = model(video_features)

            # Compute loss
            loss = criterion(outputs, labels)

            # Get the predictions
            preds = torch.argmax(outputs, dim=1)

            # Add the loss to the epoch loss  
            epoch_loss += loss.item()

            # Add the predictions and labels to the epoch predictions and labels
            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

            # Compute metrics
            for k in epoch_metrics.keys():
                epoch_metrics[k] += metrics[k](preds.cpu().numpy(), labels.cpu().numpy())

    epoch_loss /= len(dataloader)

    for k in epoch_metrics.keys():
        epoch_metrics[k] /= len(dataloader)
    
    print('eval Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))

    return epoch_loss, epoch_metrics

def plot_training(train_loss, test_loss, metrics_names, train_metrics_logs, test_metrics_logs):
    fig, ax = plt.subplots(1, len(metrics_names) + 1, figsize=((len(metrics_names) + 1) * 5, 5))

    ax[0].plot(train_loss, c='blue', label='train')
    ax[0].plot(test_loss, c='orange', label='test')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()

    for i in range(len(metrics_names)):
        ax[i + 1].plot(train_metrics_logs[i], c='blue', label='train')
        ax[i + 1].plot(test_metrics_logs[i], c='orange', label='test')
        ax[i + 1].set_title(metrics_names[i])
        ax[i + 1].set_xlabel('epoch')
        ax[i + 1].legend()

    plt.show()

def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
    for i in range(len(metrics_names)):
        curr_metric_name = metrics_names[i]
        metrics_log[i].append(new_metrics_dict[curr_metric_name])
    return metrics_log

def train_cycle(model, optimizer, criterion, metrics, train_loader, test_loader, n_epochs, device):
    train_loss_log, test_loss_log = [], []
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for _ in range(len(metrics))]
    test_metrics_log = [[] for _ in range(len(metrics))]

    for epoch in range(n_epochs):
        print("Epoch {0} of {1}".format(epoch, n_epochs))
        train_loss, train_metrics = train_epoch(model, optimizer, criterion, metrics, train_loader, device)
        test_loss, test_metrics = evaluate(model, criterion, metrics, test_loader, device)

        train_loss_log.append(train_loss)
        train_metrics_log = update_metrics_log(metrics_names, train_metrics_log, train_metrics)

        test_loss_log.append(test_loss)
        test_metrics_log = update_metrics_log(metrics_names, test_metrics_log, test_metrics)

        plot_training(train_loss_log, test_loss_log, metrics_names, train_metrics_log, test_metrics_log)

    return train_metrics_log, test_metrics_log

def f1(preds, target):
    """Compute the F1 score of the model.

    Parameters:
    -----------
    preds : list
        The predictions of the model.
    target : list
        The target values of the model.

    Returns:
    --------
    f1_score : float
        The F1 score of the model.
    """
    return f1_score(target, preds, average='macro')

def acc(preds, target):
    """Compute the accuracy of the model.

    Parameters:
    -----------
    preds : list
        The predictions of the model.
    target : list
        The target values of the model.

    Returns:
    --------
    accuracy : float
        The accuracy of the model.
    """
    return accuracy_score(target, preds)