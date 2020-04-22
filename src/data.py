import random
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

UNKNOWN_TOKEN = '<unk>'


class Document:
    def __init__(self, text, label):
        self.text = text
        self.label = label


class Corpus(Dataset):
    def __init__(self, documents):
        self.data = documents
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]
    
    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    
    def labels(self):
        return torch.LongTensor([doc.label for doc in self.data])


def get_vocabulary(vocab_path):
    vocabulary = set()
    with open(vocab_path, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            vocabulary.add(word)
    vocabulary = list(vocabulary)
    # Add token for unknown tokens
    vocabulary.append(UNKNOWN_TOKEN)
    return vocabulary


def get_labels(labels_path):
    labels = set()
    with open(labels_path, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            labels.add(word)
    return list(labels)


def get_data(data_path, all_labels):
    label_to_index = {lbl: i for i, lbl in enumerate(all_labels)}

    documents = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            label, text = line.split('\t')
            doc = Document(text.split(), label_to_index[label])
            documents.append(doc)
    
    return Corpus(documents)


def save_all_labels(out_path, train_path, test_path):
    labels = set()
    for data_path in [train_path, test_path]:
        with open(data_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                label, _ = line.split('\t')
                labels.add(label)
    labels = list(labels)
    with open(out_path, 'w+') as f:
        f.write('\n'.join(labels))


def save_vocabulary(out_path, train_path, test_path, doc_freq_threshold=5):
    word_freqs = {}  # counting num of docs
    for data_path in [train_path, test_path]:
        with open(data_path, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                _, text = line.split('\t')
                words = text.split()
                for w in set(words):
                    w = w.strip()
                    if w in word_freqs:
                        word_freqs[w] += 1
                    else:
                        word_freqs[w] = 1
    vocab = []
    for w, freq in word_freqs.items():
        if freq >= doc_freq_threshold:
            vocab.append(w)
    with open(out_path, 'w+') as f:
        f.write('\n'.join(vocab))


if __name__ == '__main__':
    import sys
    save_all_labels('data/20ng-labels.txt', 'data/train-20news.txt', 'data/test-20news.txt')
