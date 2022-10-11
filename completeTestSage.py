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



class SAGENet(torch.nn.Module):
    def __init__(self):
        super(SAGENet, self).__init__()

        self.conv = SAGEConv(dataset.num_features,
                             200,
                             aggr="max")  # max, mean, add ...)
        self.conv2 = SAGEConv(200,
                             50,
                             aggr="max")
        self.conv3 = SAGEConv(50,
                             dataset.num_classes,
                             aggr="max")

    def forward(self):
        x = self.conv(data.x, data.edge_index)
        x = self.conv2(x, data.edge_index)
        x = self.conv3(x, data.edge_index)
        return F.log_softmax(x, dim=1)


model = SAGENet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
print('===========================================================================================================')
print(model)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 100):
    train()
    _, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'

    if epoch % 10 == 0:
        print(log.format(epoch, best_val_acc, test_acc))

#model.eval()

#out = model(data.x, data.edge_index)
#visualize(out, color=data.y)