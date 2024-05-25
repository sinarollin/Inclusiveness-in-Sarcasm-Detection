import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import cv2

# Training function for 1 epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in dataloader:

        # Move batch to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss 
        loss = criterion(outputs, labels) 

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Add the loss to the epoch loss
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_preds / total_preds

    print('train Loss: {:.4f}'.format(epoch_loss),
          ', train Acc: {:.4f}'.format(epoch_acc))

    return epoch_loss, epoch_acc



# Evaluation function for 1 epoch
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in dataloader:

            # Move batch to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Add the loss to the epoch loss  
            running_loss += loss.item()

            # Get the predictions
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_preds / total_preds

    print('test Loss: {:.4f}'.format(epoch_loss),
          ', test Acc: {:.4f}'.format(epoch_acc))

    return epoch_loss, epoch_acc



# Function to extract visual features using I3D
def extract_video_features(video_path, sample_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Only add frame to list every 'sample_rate' frames
        if frame_count % sample_rate == 0:
            frame = cv2.resize(frame, (128, 128))  # Resize frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    video_tensor = torch.tensor(frames, dtype=torch.float32)  # Convert frames to tensor
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # Should be [C, T, H, W]
    
    return video_tensor



# Function to add padding to the videos
def custom_collate_fn(batch):
    videos, labels = zip(*batch)
    max_frames = max(video.size(1) for video in videos)  # Find the max number of frames

    padded_videos = []
    for video in videos:
        padding_needed = max_frames - video.size(1)
        if padding_needed > 0:
            pad = torch.zeros((video.shape[0], padding_needed, video.shape[2], video.shape[3]), dtype=video.dtype, device=video.device)
            padded_video = torch.cat([video, pad], dim=1)
        else:
            padded_video = video
        padded_videos.append(padded_video)

    videos_tensor = torch.stack(padded_videos)  # Stack along a new batch dimension
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return videos_tensor, labels_tensor


#  ----------------------------------------------------------
#  ----------------------------------------------------------

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