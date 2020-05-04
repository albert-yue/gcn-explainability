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


class SyntaxTree:
    def __init__(self, text, tree, label):
        '''
        :param text: list of strings, representing the words with the sentence
        :param tree: list of indices of corresponding parent words in the text (0-index)
            with -1 indicating the root
        '''
        self.text = text
        self.tree = tree
        self.label = label


class Treebank(Dataset):
    def __init__(self, trees):
        self.data = trees
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]
    
    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    
    def labels(self):
        return torch.FloatTensor([tree.label for tree in self.data])


def get_vocabulary(vocab_path):
    vocabulary = []
    with open(vocab_path, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            vocabulary.append(word)
    vocabulary = vocabulary
    # Add token for unknown tokens
    vocabulary.append(UNKNOWN_TOKEN)
    return vocabulary


def get_labels(labels_path):
    labels = []
    with open(labels_path, 'r') as f:
        for line in f.readlines():
            word = line.strip()
            labels.append(word)
    return labels


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


def get_treebank_data(data_path):
    '''
    Loads the Stanford Sentiment Treebank data
    Sentiment labels are between 1 and 25, where 
    '''
    pass


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
    save_all_labels('data/ohsumed-labels.txt', 'data/train-ohsumed.txt', 'data/test-ohsumed.txt')
    save_vocabulary('data/ohsumed-labels.txt', 'data/train-ohsumed.txt', 'data/test-ohsumed.txt')
