import random
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.optim import Adam


def accuracy(preds, targets):
    label_preds = torch.argmax(preds, dim=-1)
    return accuracy_score(targets, label_preds)


def evaluate(model, dataset, targets, metric_fn):
    model.eval()
    dataset_preds = []

    for example in tqdm(dataset):
        with torch.no_grad():
            preds = model(example.adj, example.inp)
        dataset_preds.append(preds)

    dataset_preds = torch.cat(dataset_preds, dim=-1)
    return metric_fn(dataset_preds, targets)


def train(model, train_dataset, val_dataset, batch_size=50, epochs=200, init_lr=0.001, early_stop_threshold=10, seed=0, val_every=1, print_every=10, plot_every=5, save_path='logs/train.pt'):
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=init_lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_with_no_improvement = 0
    for epoch in range(epochs):
        if epoch % print_every == 0:
            print('Epoch', epoch)
        
        model.train()

        train_dataset.shuffle(seed)

        epoch_loss = 0

        num_batches = len(train_dataset) // batch_size
        effective_size = num_batches * batch_size  # skip last batch for stability
        
        for i in range(0, effective_size, batch_size):
            batch = train_dataset[i:i+batch_size]

            model.zero_grad()
            for example in batch:
                out = model(example.adj, example.inp)
                loss = loss_fn(out, torch.LongTensor([example.label]))
                loss.backward()
            optim.step()
            
            epoch_loss += loss.data.item()

        train_losses.append(epoch_loss)
        if epoch % print_every == 0:
            print('Train mean cross-entropy:', loss.item())

        if epoch % val_every == 0:
            val_loss = evaluate(model, val_dataset, val_dataset.labels(), loss_fn)
            # if val_loss <= best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save(model.state_dict(), save_path)
            
            val_losses.append(val_loss.data.item())
            if epoch % print_every == 0:
                print('Validation mean cross-entropy:', val_loss.item())
            
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                epochs_with_no_improvement = 0
            else:
                epochs_with_no_improvement += 1
            
            if epochs_with_no_improvement == early_stop_threshold:
                break
    
    model.load_state_dict(torch.load(save_path))
    return train_losses, val_losses
    

if __name__ == '__main__':
    pass
