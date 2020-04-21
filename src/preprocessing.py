from collections import defaultdict
from math import log

from nltk.corpus import stopwords
import torch
import torch.sparse as sparse
from tqdm.auto import tqdm, trange

UNKNOWN_TOKEN = '<unk>'


def clean_text(corpus, vocabulary):
    stop_words = stopwords.words('english')
    for doc in tqdm(corpus):
        words = doc.text
        doc.text = [(w if w in vocabulary else UNKNOWN_TOKEN) for w in words if w not in stop_words]
    return


def build_adj_matrix(corpus, vocabulary, num_documents, doc_offset=0, window_size=20):
    """
    Builds the adjacency matrix A for the text-document graph
    
    A_ij = max(0, PMI(i,j))     if i, j are words (0 if PMI <= 0)
         = TF-IDF(i,j)          if i is a document and j is a word (or opposite)
         = 1                    if i = j
         = 0                    otherwise
    
    """
    num_words = len(vocabulary)
    num_vertices = num_words + num_documents

    word_to_index = {w: i for i, w in enumerate(vocabulary)}

    # nonzero entries in the adjacency matrix
    entries = []
    # indices for the entries
    row = []
    col = []

    print('Building word frequencies per doc')
    word_freqs_per_doc = {}
    for i, doc in enumerate(tqdm(corpus)):
        doc_idx = num_words + doc_offset + i

        word_freq = {}
        text = doc.text
        for word in text:
            if word not in vocabulary:
                word = '<unk>'
            
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        word_freqs_per_doc[doc_idx] = word_freq
    
    ### PMI calculations
    print('Building word frequencies per window')
    num_windows = 0
    word_window_occurrences = defaultdict(int)
    word_pair_window_occurrences = defaultdict(int)  # keys are tuples (x,y) where x < y are strings
    for doc in tqdm(corpus):
        text = doc.text
        for window_start_idx in range(max(1, len(text) - window_size + 1)):
            num_windows += 1
            window = text[window_start_idx:window_start_idx + window_size]
            distinct_words = list(set(window))
            for word in distinct_words:
                word_window_occurrences[word] += 1
            for i in range(len(distinct_words)):
                for j in range(i+1, len(distinct_words)):
                    word1, word2 = distinct_words[i], distinct_words[j]
                    if word2 < word1:
                        word1, word2 = word2, word1
                    key = (word1, word2)
                    word_pair_window_occurrences[key] += 1
    
    # get rid of defaultdict behavior
    word_window_occurrences = dict(word_window_occurrences)
    word_pair_window_occurrences = dict(word_pair_window_occurrences)
    
    print('Calculating PMIs')
    for pair, pair_freq in tqdm(word_pair_window_occurrences.items()):
        word1, word2 = pair
        w1_idx, w2_idx = word_to_index[word1], word_to_index[word2]
        freq1 = word_window_occurrences[word1]
        freq2 = word_window_occurrences[word2]
        # PMI = P(i and j) / (P(i) * P(j)) where P(i and j) = #windows with i and j / #windows
        # and P(i) = #windows with i / #windows
        pmi = log((pair_freq * num_windows) / (freq1 * freq2))
        
        if pmi <= 0: continue

        entries.append(pmi)
        row.append(w1_idx)
        col.append(w2_idx)

        entries.append(pmi)
        row.append(w2_idx)
        col.append(w1_idx)

    ### TF-IDF calculations
    print('Calculating TF-IDF')
    for w_idx, word in enumerate(tqdm(vocabulary)):
        doc_occurrences = 0
        for doc_idx, freqs in word_freqs_per_doc.items():
            if word in freqs:
                doc_occurrences += 1
        
        if doc_occurrences == 0: continue

        idf = log(num_documents / doc_occurrences)
        for doc_idx, freqs in word_freqs_per_doc.items():
            if word in freqs and freqs[word] > 0:
                entries.append(freqs[word] * idf)
                row.append(w_idx)
                col.append(doc_idx)

                entries.append(freqs[word] * idf)
                row.append(doc_idx)
                col.append(w_idx)

    ### 1s for identities
    print('Identities')
    for i in trange(num_vertices):
        entries.append(1)
        row.append(i)
        col.append(i)
    
    indices = torch.LongTensor([row, col])
    entries = torch.FloatTensor(entries)
    return sparse.FloatTensor(indices, entries, torch.Size([num_vertices, num_vertices]))

# def build_adj_matrix_dense(corpus, vocabulary):
#     """Builds the adjacency matrix for the text-document graph with as a regular dense tensor"""
#     # indices correspond to words in the vocabulary, then documents in order of the corpus
#     adj = torch.eye(num_vertices)
#     for i, doc in enumerate(corpus):
#         adj_idx = len(vocabulary) + i
#         text = doc.text
#         for word in text:

if __name__ == '__main__':
    from src.data import Corpus, get_data, get_vocabulary, get_labels
    vocab = get_vocabulary('data/20ng-vocabulary.txt')
    labels = get_labels('data/20ng-labels.txt')
    corpus = get_data('data/test.txt', labels)
    clean_text(corpus, vocab)
    adj = build_adj_matrix(corpus, vocab)
    print(adj)
