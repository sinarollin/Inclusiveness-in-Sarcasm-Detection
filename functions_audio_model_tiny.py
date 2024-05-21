

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.metrics import Precision, Recall

def train_cycle(model, optimizer, loss_fn, train_acc_metric, val_acc_metric, train_dataloader, val_dataloader, epochs):
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []

    precision = Precision()
    recall = Recall()

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        epoch_loss = 0

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataloader):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)
            precision.update_state(y_batch_train, tf.argmax(logits, axis=-1))
            recall.update_state(y_batch_train, tf.argmax(logits, axis=-1))
            epoch_loss += loss_value

        train_f1 = 2 * ((precision.result() * recall.result()) / (precision.result() + recall.result() + tf.keras.backend.epsilon()))
        train_f1s.append(train_f1)
        train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(train_loss)
        train_acc = train_acc_metric.result()
        train_accs.append(float(train_acc))
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print("Training F1 score over epoch: %.4f" % (float(train_f1),))
        print("Training loss over epoch: %.4f" % (float(train_loss),))

        # Run a validation loop at the end of each epoch.
        val_epoch_loss = 0
        for x_batch_val, y_batch_val in val_dataloader:
            val_logits = model(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_logits)
            val_epoch_loss += val_loss_value

            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
            precision.update_state(y_batch_val, tf.argmax(val_logits, axis=-1))
            recall.update_state(y_batch_val, tf.argmax(val_logits, axis=-1))
            
        val_f1 = 2 * ((precision.result() * recall.result()) / (precision.result() + recall.result() + tf.keras.backend.epsilon()))
        val_f1s.append(val_f1)
        val_loss = val_epoch_loss / len(val_dataloader)
        val_losses.append(val_loss)
        val_acc = val_acc_metric.result()
        val_accs.append(float(val_acc))
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation F1 score: %.4f" % (float(val_f1),))
        print("Validation loss: %.4f" % (float(val_loss),))

    return train_accs, val_accs, train_losses, val_losses, train_f1s, val_f1s


def plot_metrics(train_accs, val_accs, train_losses, val_losses, train_f1s, val_f1s):
    epochs = range(len(train_accs))

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1s, label='Training F1 Score')
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.legend(loc='lower right')
    plt.title('Training and Validation F1 Score')

    

    plt.tight_layout()
    plt.show()