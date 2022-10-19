import os
from tqdm import trange
import torch
import random
import numpy as np
os.environ['TORCH'] = torch.__version__
print(torch.__version__)


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Testing node label rate: {(data.test_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

gt = data.y.tolist()
train = []
test = []
val = []
list(set(gt))
for label in list(set(gt)):
    output = [idx for idx, element in enumerate(gt) if element == label]
    #shuffle output?
    random.shuffle(output)
    train_p, val_p, test_p = np.split(output, [int(len(output) * 0.1), int(len(output) * 0.3)])
    train.extend(train_p), val.extend(val_p), test.extend(test_p)

trainMask = [False]*gt.__len__()
valMask = [False]*gt.__len__()
testMask = [False]*gt.__len__()
for i in train:
    trainMask[i] = True
for i in val:
    valMask[i] = True
for i in test:
    testMask[i] = True

data.train_mask = torch.BoolTensor(trainMask)
data.val_mask = torch.BoolTensor(valMask)
data.test_mask = torch.BoolTensor(testMask)

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
import torch
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        self.conv1bis = GCNConv(hidden_channels2, hidden_channels2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv1bis(x, edge_index)
        #x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCN2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, 200)
        self.conv1bis = GCNConv(200, 100)
        self.conv2bis = GCNConv(100, 40)
        self.conv2 = GCNConv(40, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1bis(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2bis(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x



#model = GCN(hidden_channels=16)
#print(model)

#model = GCN(hidden_channels=16)
#model.eval()

#out = model(data.x, data.edge_index)
#visualize(out, color=data.y)



model = GCN2()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
print('===========================================================================================================')
print(model)

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc


for epoch in trange(1, 201):
    loss = train()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

#model.eval()

#out = model(data.x, data.edge_index)
#visualize(out, color=data.y)