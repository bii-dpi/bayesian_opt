import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, init_dim, hidden_dim_gcn, output_dim, dropout_prop, hidden_dim_fc, num_classes):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(init_dim, hidden_dim_gcn)
        self.gc2 = GraphConvolution(hidden_dim_gcn, output_dim)
        self.dropout_prop = dropout_prop
        self.fc1 = nn.Linear(output_dim, hidden_dim_fc)
        self.fc2 = nn.Linear(hidden_dim_fc, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout_prop, training=self.training)
        x = F.relu(self.gc2(x, adj))
        # NEED TO FLATTEN SOMEWHERE
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x), dim=1)
