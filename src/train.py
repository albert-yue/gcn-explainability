import random
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.optim import Adam

from src.data import Corpus, get_data, get_vocabulary, get_labels
from src.preprocessing import clean_text, build_adj_matrix, normalize_adj


def accuracy(preds, targets):
    label_preds = torch.argmax(preds, dim=-1)
    return accuracy_score(targets, label_preds)


def evaluate(model, adj_matrix, targets, metric_fn, start_idx=None, end_idx=None):
    model.eval()
    with torch.no_grad():
        preds = model(adj_matrix)
    
    if start_idx is None: start_idx = 0
    if end_idx is None: end_idx = preds.size(0)
    preds = preds[start_idx:end_idx, :]

    return metric_fn(preds, targets)


def train(model, train_adj_matrix, val_adj_matrix, train_labels, val_labels, vocab_size, epochs=200, init_lr=0.02, early_stop_threshold=10, val_every=1, print_every=10, plot_every=5, save_path='logs/train.pt'):
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=init_lr)

    train_size, val_size = len(train_labels), len(val_labels)
    val_start_idx = vocab_size + train_size
    test_start_idx = vocab_size + train_size + val_size

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_with_no_improvement = 0
    for epoch in range(epochs):
        if epoch % print_every == 0:
            print('Epoch', epoch)
        
        model.train()
        model.zero_grad()

        out = model(train_adj_matrix)
        out = out[vocab_size:val_start_idx, :]
        loss = loss_fn(out, train_labels)
        loss.backward()
        optim.step()

        train_losses.append(loss.data.item())
        if epoch % print_every == 0:
            print('Train mean cross-entropy:', loss.item())

        if epoch % val_every == 0:
            val_loss = evaluate(model, val_adj_matrix, val_labels, loss_fn, start_idx=val_start_idx, end_idx=test_start_idx)
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
    from src.data import Corpus, get_data, get_vocabulary, get_labels
    from src.preprocessing import clean_text, build_adj_matrix
    from src.models.gcn import GCN
    
    seed = 0
    val_split = 0.1

    vocab = get_vocabulary('data/20ng-vocabulary.txt')
    labels = get_labels('data/20ng-labels.txt')
    corpus = get_data('data/train-20news.txt', labels)
    test_corpus = get_data('data/test-20news.txt', labels)

    # Mask out unknown words
    clean_text(corpus, vocab)
    clean_text(test_corpus, vocab)

    # Split validation set
    corpus.shuffle(seed)
    len_train = int(len(corpus) * (1 - val_split))
    train_corpus = Corpus(corpus[:len_train])
    val_corpus = Corpus(corpus[len_train:])

    num_documents = len(train_corpus) + len(val_corpus) + len(test_corpus)
    train_adj_matrix = build_adj_matrix(train_corpus, vocab, num_documents, doc_offset=0)
    val_adj_matrix = build_adj_matrix(val_corpus, vocab, num_documents, doc_offset=len(train_corpus))
    test_adj_matrix = build_adj_matrix(test_corpus, vocab, num_documents, doc_offset=len(train_corpus) + len(val_corpus))

    train_adj_matrix = normalize_adj(train_adj_matrix)
    val_adj_matrix = normalize_adj(val_adj_matrix)
    test_adj_matrix = normalize_adj(test_adj_matrix)

    hidden_size = 200  # hyperparameter
    dropout = 0.5  # hyperparameter
    epochs = 500

    num_vertices = len(vocab) + num_documents
    model = GCN(num_vertices, hidden_size, len(labels), len(vocab), dropout=dropout)

    print('Start training')
    train_losses, val_losses = train(model, train_adj_matrix, val_adj_matrix, train_corpus.labels(), val_corpus.labels(), len(vocab), epochs=epochs, early_stop_threshold=10)
    print(train_losses)
    print(val_losses)

    test_start_idx = vocab_size + train_size + val_size
    test_loss = evaluate(model, test_adj_matrix, test_corpus.labels(), nn.CrossEntropyLoss(), start_idx=test_start_idx)
    print(test_loss)
