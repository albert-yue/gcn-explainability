'''
Code for processing data from the Stanford Sentiment Treebank dataset into the input format for a GCN
'''

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def load_glove_embeddings(filepath):
    mapping = {}
    with open(filepath, 'r') as f:
        for line in tqdm(f.readlines()):
            word, vec = line.split(maxsplit=1)
            vec = vec.strip()
            vec = torch.FloatTensor([[float(f) for f in vec.split()]])
            mapping[word.lower()] = vec
    return mapping


def generate_unknown_embedding():
    return 


class LabelledBinaryParseTree:
    def __init__(self, text=None, label=None, children=None):
        self.text = text or ''
        self.label = label
        self.parent = None
        self.children = children or []
    
    def add_child(self, node):
        self.children.append(node)
        node.parent = self
    
    def _repr_helper(self, indent=0):
        if self.label is not None and len(self.text) > 0:
            text = '{} {}'.format(self.label, self.text)
        else:
            text = '{}{}'.format(self.label if self.label is not None else '', self.text)
        indent_str = ' ' * indent
        
        if len(self.children) == 0:
            return indent_str + '(' + text + ')'
        
        children_repr = [c._repr_helper(indent+2) for c in self.children]
        return '{ind}({t} \n{chs}\n{ind})'.format(ind=indent_str, t=text, chs='\n'.join(children_repr))
    
    def __repr__(self):
        return self._repr_helper()


def add_label_text(node, data):
    split_data = data.split(maxsplit=1)
    node.label = int(split_data[0])
    if len(split_data) > 1:
        node.text = split_data[1]


def parse_tree(line):
    root = None
    curr_node = None
    curr_word = ''
    depth = 0
    for char in line:
        if char == '(':
            # add curr_word, if any, to curr_node.text
            if len(curr_word.strip()) > 0:
                add_label_text(curr_node, curr_word.strip())

            # spawn child node that's just started
            if root is None:
                root = LabelledBinaryParseTree()
                curr_node = root
            else:
                new_node = LabelledBinaryParseTree()
                curr_node.add_child(new_node)
                curr_node = new_node
            depth += 1
            curr_word = ''
        elif char == ')':
            if len(curr_word.strip()) > 0:
                add_label_text(curr_node, curr_word.strip())
                curr_word = ''
            curr_node = curr_node.parent
            depth -= 1
        else:
            curr_word += char
    
    if depth != 0:
        raise ValueError('Number of opening and closing parentheses does not match.')
    
    return root


def extract_tokens(root):
    if len(root.children) == 0:
        return [root.text]
    
    left = extract_tokens(root.children[0])
    right = extract_tokens(root.children[-1])
    return left + [root.text] + right


def extract_sentence(root):
    return ' '.join(t for t in extract_tokens(root) if len(t) > 0)


def augment_tree_metadata(root):
    if len(root.children) == 0:
        root.left_word = root.text if len(root.text) > 0 else None
        root.right_word = root.text if len(root.text) > 0 else None
        root.size = 1
        return
    
    if len(root.children) == 1:
        raise ValueError('Unbalanced binary tree')
    
    for child in root.children:
        augment_tree_metadata(child)
    
    root.left_word = root.children[0].left_word if len(root.children[0].left_word) > 0 else None
    root.right_word = root.children[-1].right_word if len(root.children[-1].right_word) > 0 else None
    root.size = sum(c.size for c in root.children) + 1


def extract_parent_ptrs(root, index_offset=0, parent=-1):
    '''
    :param index_offset: lowest possible index for all nodes in the tree under `root`
    '''
    if len(root.children) == 0:
        return [parent]
    
    left = root.children[0]
    right = root.children[-1]
    
    index = left.size + index_offset
    
    return extract_parent_ptrs(left, index_offset=index_offset, parent=index) + [parent] + extract_parent_ptrs(right, index_offset=index+1, parent=index)


def extract_embeddings(root, embedding_map):
    if len(root.text) > 0:
        root_embedding = embedding_map[root.text.lower()]
    else:
        root_embedding = embedding_map[root.left_word.lower()] + embedding_map[root.right_word.lower()]

    if len(root.children) == 0:
        return [root_embedding]
    
    left = extract_embeddings(root.children[0], embedding_map)
    right = extract_embeddings(root.children[-1], embedding_map)
    return left + [root_embedding] + right


def extract_adj_matrix_and_input(root, embedding_map):
    augment_tree_metadata(root)
    
    embeddings = extract_embeddings(root, embedding_map)
    inp = torch.cat(embeddings, dim=0)

    parent_ptrs = extract_parent_ptrs(root)
    num_vertices = len(parent_ptrs)
    adj = torch.zeros(num_vertices, num_vertices)
    for i, ptr in enumerate(parent_ptrs):
        if ptr != -1:
            adj[i, ptr] = 1
            adj[ptr, i] = 1
    return adj, inp


class GCNInput:
    def __init__(self, adj, inp, label):
        self.adj = adj
        self.inp = inp
        self.label = label


class GCNDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]
    
    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    
    def labels(self):
        return torch.FloatTensor([ex.label for ex in self.data])


if __name__ == '__main__':
    import pickle
    from src.data import Treebank, SyntaxTree

    embedding_map = load_glove_embeddings('data/glove.6B.200d.txt')
    trees = []
    with open('data/stanfordSentimentTreebank/dev.txt') as f:
        for line in tqdm(f.readlines()):
            tree = parse_tree(line)
            sentence = extract_sentence(tree)
            label = tree.label
            trees.append(SyntaxTree(sentence, tree, label))
    treebank = Treebank(trees)

    inputs = []
    for example in treebank:
        tree = example.tree
        adj, inp = extract_adj_matrix_and_input(tree, embedding_map)
        inputs.append(GCNInput(adj, inp, example.label))
    dataset = GCNDataset(inputs)
    torch.save(dataset, 'stanford_dev.pt')
