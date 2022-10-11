import torch
from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for d in dataset:
    print(d.num_nodes)

for data in loader:
    print(data)


    print(data.num_graphs)

    x = scatter_mean(data.x, data.batch, dim=0)
    x.size()
    print(torch.Size([32, 21]))