import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, HeteroConv

class WeightedHierarchicalGNN(torch.nn.Module):
    def __init__(self, num_l4_features, num_l5_features, hidden_channels, num_classes=1, model_type='GCN'):
        super(WeightedHierarchicalGNN, self).__init__()
        self.model_type = model_type
        
        def make_conv(in_channels, out_channels, weighted=False):
            if weighted or model_type == 'GCN':
                return GCNConv(in_channels, out_channels)
            else:
                return SAGEConv(in_channels, out_channels)

        self.conv1 = HeteroConv({
            
            ('l4', 'correlated', 'l4'): GCNConv(num_l4_features, hidden_channels),
            
            ('l4', 'therapeutic', 'l4'): make_conv(num_l4_features, hidden_channels),
            
            ('l5', 'part_of', 'l4'): SAGEConv((num_l5_features, num_l4_features), hidden_channels),
            
            ('l4', 'has_part', 'l5'): SAGEConv((num_l4_features, num_l5_features), hidden_channels),
        }, aggr='sum')
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = HeteroConv({
            ('l4', 'correlated', 'l4'): GCNConv(hidden_channels, hidden_channels),
            ('l4', 'therapeutic', 'l4'): make_conv(hidden_channels, hidden_channels),
            ('l5', 'part_of', 'l4'): SAGEConv(hidden_channels, hidden_channels),
            ('l4', 'has_part', 'l5'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')
        
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.lin_l4 = nn.Linear(hidden_channels, num_classes)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None, return_embedding=False):
        
        x_out = {}
        
        out_dict = {}
        
        def get_kwargs(edge_type):
            kwargs = {}
            if edge_weight_dict and edge_type in edge_weight_dict:
                kwargs['edge_weight'] = edge_weight_dict[edge_type]
            return kwargs

        for edge_type, conv in self.conv1.convs.items():
            src, rel, dst = edge_type
            if src not in x_dict: continue 
            
            x_src = x_dict[src]
            
            edge_index = edge_index_dict.get(edge_type)
            
            if edge_index is not None:
                if isinstance(conv, SAGEConv):
                    
                    if src != dst:
                        if dst in x_dict:
                            x_dst = x_dict[dst]
                            out = conv((x_src, x_dst), edge_index)
                        else:
                             
                             continue
                    else:
                        out = conv(x_src, edge_index)
                else:
                    
                    kwargs = get_kwargs(edge_type)
                    out = conv(x_src, edge_index, **kwargs)
                
                if dst not in out_dict:
                    out_dict[dst] = []
                out_dict[dst].append(out)
                
        for node_type, outs in out_dict.items():
             x_out[node_type] = sum(outs) 
             
        x_dict = {key: self.bn1(F.relu(x)) for key, x in x_out.items()}
        x_dict = {key: F.dropout(x, p=0.5, training=self.training) for key, x in x_dict.items()}
        
        out_dict = {}
        for edge_type, conv in self.conv2.convs.items():
            src, rel, dst = edge_type
            if src not in x_dict: continue 
            
            x_src = x_dict[src]
            
            edge_index = edge_index_dict.get(edge_type)
            if edge_index is not None:
                kwargs = get_kwargs(edge_type)
                if isinstance(conv, SAGEConv):
                     if src != dst and dst in x_dict:
                         x_dst = x_dict[dst]
                         out = conv((x_src, x_dst), edge_index)
                     else:
                        out = conv(x_src, edge_index)
                else:
                    out = conv(x_src, edge_index, **kwargs)
                
                if dst not in out_dict:
                    out_dict[dst] = []
                out_dict[dst].append(out)
        
        x_out = {key: sum(outs) for key, outs in out_dict.items()}
        
        x_dict = {key: self.bn2(F.relu(x)) for key, x in x_out.items()}
        x_dict = {key: F.dropout(x, p=0.5, training=self.training) for key, x in x_dict.items()}
        
        if return_embedding:
            return self.lin_l4(x_dict['l4']), x_dict['l4']
        return self.lin_l4(x_dict['l4'])