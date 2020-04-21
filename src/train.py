import random
import torch
import torch.nn as nn
from torch.optim import Adam

from src.data import Corpus, get_data, get_vocabulary, get_labels
from src.preprocessing import mask_unknown_words, build_adj_matrix


def evaluate(model, adj_matrix, targets, metric_fn):
    model.eval()
    with torch.no_grad():
        preds = model(adj_matrix)
    return metric_fn(preds, targets)


def train(model, corpus, epochs=100, init_lr=0.001, val_split=0.1, val_every=1, seed=None, print_every=10, plot_every=5, save_path='logs/train.pt'):
    # Split validation set
    corpus.shuffle(seed)
    len_train = int(len(corpus) * (1 - val_split))
    train_corpus = Corpus(corpus[:len_train])
    val_corpus = Corpus(corpus[len_train:])

    train_adj_matrix = build_adj_matrix(train_corpus, vocabulary)
    val_adj_matrix = build_adj_matrix(val_corpus, vocabulary)

    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=init_lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        if epoch % print_every == 0:
            print('Epoch', epoch)
        
        model.train()
        model.zero_grad()

        out = model(train_adj_matrix)
        loss = loss_fn(out, train_corpus.labels())
        loss.backward()
        optim.step()

        train_losses.append(loss.data.item())
        if epoch % print_every == 0:
            print('Train mean cross-entropy:', val_loss)

        if epoch % val_every == 0:
            val_loss = evaluate(model, val_adj_matrix, val_corpus.labels(), loss_fn)
            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), save_path)
            
            val_losses.append(val_loss.data.item())
            if epoch % print_every == 0:
                print('Validation mean cross-entropy:', val_loss)
    
    model.load_state_dict(torch.load(save_path))
    return train_losses, val_losses
    

if __name__ == '__main__':
    seed = 0

    from src.data import Corpus, get_data, get_vocabulary, get_labels
    vocab = get_vocabulary('data/20ng-vocabulary.txt')
    labels = get_labels('data/20ng-labels.txt')
    corpus = get_data('data/train-20news.txt', labels)
    test_corpus = get_data('data/test-20news.txt', labels)

    from src.preprocessing import mask_unknown_words, build_adj_matrix

    # Mask out unknown words
    mask_unknown_words(corpus, vocab)
    mask_unknown_words(test_corpus, vocab)

    from src.models.gcn import GCN
    hidden_size = 200  # hyperparameter
    dropout = 0.5  # hyperparameter

    num_vertices = len(vocab) + len(corpus)
    model = GCN(num_vertices, hidden_size, 2, len(vocab), dropout=dropout)

    train_losses, val_losses = train(model, corpus, epochs=10, seed=seed, plot_every=1)
    print(train_losses)
    print(val_losses)

    test_adj = build_adj_matrix(test_corpus, vocab)
    test_loss = evaluate(model, test_adj, test_corpus.labels(), nn.CrossEntropyLoss())
    print(test_loss)
