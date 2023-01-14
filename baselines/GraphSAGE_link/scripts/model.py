
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing, SAGEConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

class GraphSage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args, bias=True):
        super(GraphSage, self).__init__()

        self.num_layers = args.num_layers
        self.dropout_p = args.dropout_p
        self.use_gpu = args.use_gpu
        self.use_multiple_gpu = args.use_multiple_gpu
        self.device = args.device

        self.convs = torch.nn.ModuleList()
        if self.use_gpu is True:
            for layer in range(self.num_layers):
                if self.use_multiple_gpu is True:
                    device = f"cuda:{layer % torch.cuda.device_count()}"
                else:
                    device = self.device
                if layer == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean', bias=bias).to(device))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean', bias=bias).to(device))
            layer = layer + 1
            if self.use_multiple_gpu is True:
                device = f"cuda:{layer % torch.cuda.device_count()}"
            else:
                device = self.device
            self.lin = torch.nn.Linear(hidden_channels*2,out_channels,bias=bias).to(device)
        else:
            for layer in range(self.num_layers):
                if layer == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean', bias=bias))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean', bias=bias))
            self.lin = torch.nn.Linear(hidden_channels*2,out_channels,bias=bias)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adjs, link, n_id):

        if self.use_gpu is True:

            for i, (edge_index, size) in enumerate(adjs):
                if self.use_multiple_gpu is True:
                    device = f"cuda:{i % torch.cuda.device_count()}"
                else:
                    device = self.device
                x = x.to(device)
                x_target = x[:size[1]].to(device)
                edge_index = edge_index.to(device)
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
            i = i + 1
            if self.use_multiple_gpu is True:
                device = f"cuda:{i % torch.cuda.device_count()}"
            else:
                device = self.device
            if link.shape[0]==1:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]]].view(1,-1),x[[torch.where(n_id==i)[0][0] for i in link[:,1]]].view(1,-1)], dim=1)
            else:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]],:],x[[torch.where(n_id==i)[0][0] for i in link[:,1]],:]], dim=1)
            x = x.to(device)
            x = self.lin(x).squeeze(-1) 

        else:
            for i, (edge_index, size) in enumerate(adjs):
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)

            if link.shape[0]==1:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]]].view(1,-1),x[[torch.where(n_id==i)[0][0] for i in link[:,1]]].view(1,-1)], dim=1)
            else:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]],:],x[[torch.where(n_id==i)[0][0] for i in link[:,1]],:]], dim=1)
            x = self.lin(x).squeeze(-1)   

        return x

    def get_gnn_embedding(self, x, adjs, n_id):

        if self.use_gpu is True:

            for i, (edge_index, size) in enumerate(adjs):
                if self.use_multiple_gpu is True:
                    device = f"cuda:{i % torch.cuda.device_count()}"
                else:
                    device = self.device
                x = x.to(device)
                x_target = x[:size[1]].to(device)
                edge_index = edge_index.to(device)
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)

            return x
        else:
            for i, (edge_index, size) in enumerate(adjs):
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)

            return x


    def predict1(self, x, adjs, link, n_id):
        res = torch.sigmoid(self.forward(x, adjs, link, n_id)).cpu().detach()
        
        return torch.stack([1-res, res],dim=1)


    def predict2(self, x, link, n_id):
        
        if self.use_gpu is True:

            if link.shape[0]==1:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]]].view(1,-1),x[[torch.where(n_id==i)[0][0] for i in link[:,1]]].view(1,-1)], dim=1)
            else:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]],:],x[[torch.where(n_id==i)[0][0] for i in link[:,1]],:]], dim=1)
            x = x.to(self.device)
            return torch.sigmoid(self.lin(x).squeeze(-1)).cpu().detach()

        else:

            if link.shape[0]==1:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]]].view(1,-1),x[[torch.where(n_id==i)[0][0] for i in link[:,1]]].view(1,-1)], dim=1)
            else:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]],:],x[[torch.where(n_id==i)[0][0] for i in link[:,1]],:]], dim=1)
            return torch.sigmoid(self.lin(x).squeeze(-1)).cpu().detach()

