import math
import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.FloatTensor(input_size, output_size))
        self.reset_parameters()
    
    def reset_parameters(self):
        # same param as nn.Linear.weight's reset_parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def __repr__(self):
        return 'GraphConv(input_size={}, output_size={})'.format(self.input_size, self.output_size)
    
    def forward(self, adj_matrix, inp=None):
        """
        Forward step: AXW
        :param adj_matrix: A tensor of size (num_vertices, num_vertices) representing
            the adjacency matrix of the text-document graph where num_vertices is the 
            number of vertices in the graph
        :param inp: A tensor of size (num_vertices, input_size) representing the input X 
            Default: `None`, where inp is assumed to be the identity
        """
        ax = adj_matrix
        if inp is not None:  # i.e. not identity
            ax = adj_matrix.matmul(inp)
        return ax.matmul(self.weight)



class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.):
        super(GCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.dropout = nn.Dropout(dropout)
        self.act_func = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

        self.layer1 = GraphConv(input_size, hidden_size)
        self.layer2 = GraphConv(hidden_size, output_size)

    
    def forward(self, adj_matrix, inp=None):
        out = self.act_func(self.layer1(adj_matrix, inp))
        out = self.dropout(out)
        out = self.layer2(adj_matrix, out)
        
        if not self.training:
            out = self.softmax(out)
        
        return out
