import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    # TODO: Implement the training loop
    
    history = []

    # optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=0.0)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    n = X_train.shape[0]
    n_eval = X_val.shape[0]

    for epoch in range(epochs):
        indices = torch.randperm(n)
        X_epoch = X_train[indices]
        y_epoch = y_train[indices]

        train_losses = []
        
        model.train()
        for batch_start in range(0, n, batch_size):
            X_batch = X_epoch[batch_start: min(batch_start + batch_size, n)]
            y_batch = y_epoch[batch_start: min(batch_start + batch_size, n)]

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach())

        # Validation loop -- no need to shuffle
        model.eval()
        val_losses = []
        val_accuracies = []
        with torch.no_grad():
            for batch_start in range(0, n_eval, batch_size):
                X_batch = X_val[batch_start: min(batch_start + batch_size, n)]
                y_batch = y_val[batch_start: min(batch_start + batch_size, n)]

                out = model(X_batch) # logits
                loss = criterion(out, y_batch)
                accuracy = (y_batch == out.argmax(dim=-1)).float().mean()

                val_losses.append(loss.detach())
                val_accuracies.append(accuracy.detach())
        
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": torch.stack(train_losses).mean(),
                "val_loss": torch.stack(val_losses).mean(),
                "val_accuracy": torch.stack(val_accuracies).mean()
            }
        )
    
    return history
