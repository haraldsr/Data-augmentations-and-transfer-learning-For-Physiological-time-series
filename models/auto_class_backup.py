import torch
import torch.nn as nn
import numpy as np
from torch.fft import fft

class Attention(nn.Module):
    
    def __init__(self, hid_dim, d):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
        self.W = nn.Linear(d, self.hid_dim, bias=True) # should be initialized with glorot_uniform
        self.u = nn.Linear(self.hid_dim, 1, bias=False) # should be initialized with glorot_uniform
        self.softmax = nn.Softmax(-2)
        self.reset_parameters()
        
    def forward(self, x, mask=None, mask_value=-1e30):
        if not mask:
            mask = torch.ones([x.shape[0], x.shape[1]], device=x.device)
        attn_weights = self.u(torch.tanh(self.W(x)))
        mask = mask.unsqueeze(-1)
        attn_weights = mask*attn_weights + (1-mask)*mask_value
        attn_weights = self.softmax(attn_weights)
        return attn_weights
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.u.weight)
        torch.nn.init.zeros_(self.W.bias)
    
    def __repr__(self):
        return f"Attention(hid_dim={self.hid_dim}, d={self.W.in_features})"
    
    def to(self, device):
        self.W.to(device)
        self.u.to(device)
        return self
    
    def __str__(self):
        return f"Attention(hid_dim={self.hid_dim}, d={self.W.in_features})"
    
    def state_dict(self):
        return {
            'W': self.W.state_dict(),
            'u': self.u.state_dict(),
            'softmax': self.softmax.state_dict()
        }

class CNNAttnModel(nn.Module):
    def __init__(self, dropout: float, nodes: int, numfeats: int, R_U: bool = False, kernelsize: list = [3], numLayers: int = 1, attention: bool = True):
        super(CNNAttnModel, self).__init__()
        self.attention = attention
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(numfeats, nodes, kernel_size=kernelsize[0], padding=1),
            nn.BatchNorm1d(nodes),
            nn.LeakyReLU(),
            #nn.AdaptiveMaxPool1d(output_size=nodes),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=dropout)
        )
        n_h = numLayers
        if (n_h) % 2 == 0:
            n_h += 1
            middle = n_h // 2
            centered_list = nodes * np.array(list(range(1, middle + 1)) + list(range(middle, 0, -1)))
        else:
            middle = n_h // 2
            centered_list = nodes * np.array(list(range(1, middle + 1)) + [middle+1] + list(range(middle, 0, -1)))
        for i in range(1, numLayers):
            if R_U:
                self._add_layer(centered_list[i-1], dropout, kernelsize[i], i)
            else:
                self._add_layer(nodes, dropout, kernelsize[i], i)
        if attention:
            self.attn = Attention(2*nodes, nodes)
    
    def _add_layer(self, nodes: int, dropout: float, kernelsize: int, layer_index: int):
        self.cnn_layers.add_module("conv_" + str(layer_index), nn.Conv1d(nodes, nodes, kernel_size=kernelsize, padding=1))
        self.cnn_layers.add_module("batchnorm_" + str(layer_index), nn.BatchNorm1d(nodes))
        self.cnn_layers.add_module("lrelu_" + str(layer_index), nn.LeakyReLU())
        #self.cnn_layers.add_module("maxpool_" + str(layer_index), nn.AdaptiveMaxPool1d(output_size=nodes))
        self.cnn_layers.add_module("maxpool_" + str(layer_index), nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.cnn_layers.add_module("dropout_" + str(layer_index), nn.Dropout(p=dropout))
    
    def forward(self, data):
        out = self.cnn_layers(data.transpose(1, 2))
        if self.attention:
            attn_weights = self.attn(out.transpose(1, 2))
            out = torch.sum(out.transpose(1, 2) * attn_weights, dim=-2)
            return out
        else:
            return out.mean(axis=-1)
    
    #def load(self, checkpath):
    #    self.load_state_dict(torch.load(checkpath))

class Linear(nn.Module):
    def __init__(self, input_size_all, hidden_size, out_size):
        super(Linear, self).__init__()
        self.firstLinear = nn.Linear(sum(input_size_all), hidden_size)
        self.finalLinear = nn.Linear(hidden_size, out_size)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.LReLU = nn.LeakyReLU()
        Ms = []
        for i in range(len(input_size_all)):
            if input_size_all[i] > 0:
                ModelLinear = nn.Linear(input_size_all[i], out_size)
                Ms.append(ModelLinear)
        self.CompOuts = nn.ModuleList(Ms)

    def forward(self, out):
        cont_emb = torch.cat(out, -1)
        op = self.LReLU(self.firstLinear(cont_emb))
        op = self.batchnorm(op)
        op = self.finalLinear(op).squeeze(-1)

        return op

class Autoencoder(nn.Module):
    def __init__(self, dropout, nodes_per_freq, nodes_before_last_layer, numfreqs, NumFeats, architecture, config=None):
        super(Autoencoder, self).__init__()
        self.numfreqs = numfreqs
        j = 0
        self.modelindx = {}
        Model_per_freq = []
        for i in range(numfreqs):
            self.modelindx[i] = j
            if nodes_per_freq[i] != 0:
                if architecture == 'CNN':
                    Model_per_freq.append(CNNAttnModel(dropout, nodes_per_freq[i], NumFeats[i], R_U=config.R_U, kernelsize=config.CNNKernelSize, numLayers=config.NumLayers, attention=False))
                elif architecture == 'CNNAttn':
                    Model_per_freq.append(CNNAttnModel(dropout, nodes_per_freq[i], NumFeats[i], R_U=config.R_U, kernelsize=config.CNNKernelSize, numLayers=config.NumLayers))
                else:
                    raise ValueError(f'No model named: {architecture}')
                j += 1

        self.freqmodels = nn.ModuleList(Model_per_freq)
        self.nodes_per_freq = nodes_per_freq
        Out_Nodes = config.Out_Nodes
        
        self.linear = Linear(nodes_per_freq, nodes_before_last_layer, Out_Nodes)
    
    def forward(self, data):
        out = [self.freqmodels[self.modelindx[i]](data[i]) for i in range(self.numfreqs) if self.nodes_per_freq[i] != 0]

        op = self.linear(out)
        return op

class Classification(nn.Module):
    def __init__(self, trained_model, nodes_before_last_layer, NumClasses):
        super(Classification, self).__init__()
        self.freqmodels = trained_model.freqmodels
        self.numfreqs = trained_model.numfreqs
        self.nodes_per_freq = trained_model.nodes_per_freq
        self.modelindx = trained_model.modelindx
        self.linear = Linear(self.nodes_per_freq, nodes_before_last_layer, NumClasses)

        #self.Finallayer = nn.Sigmoid() if NumClasses == 1 else nn.Softmax(dim=1)
            

    def forward(self, data):
        out = [self.freqmodels[self.modelindx[i]](data[i]) for i in range(self.numfreqs) if self.nodes_per_freq[i] != 0]

        op = self.linear(out)

        #op = self.Finallayer(op)
        return op
    
class Per_Sub_selfsup(nn.Module):
    def __init__(self, trained_model, nodes_before_last_layer, NumClasses):
        super(Per_Sub_selfsup, self).__init__()
        self.freqmodels = trained_model.freqmodels
        self.numfreqs = trained_model.numfreqs
        self.nodes_per_freq = trained_model.nodes_per_freq
        self.modelindx = trained_model.modelindx
        self.linear = Linear(self.nodes_per_freq, nodes_before_last_layer, NumClasses)

    def forward(self, data):
        out = [self.freqmodels[self.modelindx[i]](data[i]) for i in range(self.numfreqs) if self.nodes_per_freq[i] != 0]
        op = self.linear(out)
        return op
    
def getModel(config: object, trained_model = None, NumClasses: int = 1, nodes_before_last_layer: int = 500, classification: bool = True) -> object:
    """
    Get the model based on the given configuration.

    Parameters:
        config (object): The configuration object.
        trained_model (torch.model, optional): The trained model to use. Defaults to None.
        NumClasses (int, optional): The number of classes. Defaults to 1.
        nodes_before_last_layer (int, optional): The number of nodes before the last layer. Defaults to 500.
        classification (bool, optional): Whether it is a classification model. Defaults to True.

    Returns:
        object: The model.
    """
    if trained_model is None:
        model = Autoencoder(config.dropout, nodes_per_freq=config.nodes_per_freq, nodes_before_last_layer=config.nodes_before_last_layer, architecture=config.architecture, numfreqs=config.NumComps, NumFeats=config.NumFeats, config=config)
    else:
        if classification:
            model = Classification(trained_model, nodes_before_last_layer=nodes_before_last_layer, NumClasses=NumClasses)
            #model = Classification(trained_model, nodes_before_last_layer=int(sum(config.nodes_per_freq)/2), NumClasses=NumClasses)
        else:
            model = Per_Sub_selfsup(trained_model, nodes_before_last_layer=nodes_before_last_layer, NumClasses=NumClasses)
            
    return model