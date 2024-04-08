import pickle
import math
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import dgl
import numpy as np
import torch as th
from dgl.nn import GraphConv, GATConv
import warnings
warnings.filterwarnings("ignore")


# Feature Path
Feature_Path = "./Feature/"
# Seed
SEED = 1047
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    # torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)

# model parameters
ADD_NODEFEATS = 'all'  # all/atom_feats/psepose_embedding/no
MAP_CUTOFF = 6
# MAP_CUTOFF = 14 best 15 best
DIST_NORM = 15

# INPUT_DIM
INPUT_DIM = 2614+512-512
HIDDEN_DIM = 32  
DROPOUT = 0.5
LEARNING_RATE = 1E-3
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2  
NUMBER_EPOCHS = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embedding(sequence_name):
    pssm_feature = np.load( "../实验数据2/PSSM_npy/" + sequence_name + '.npy')
    esm_feature = np.load( "../实验数据2/ESM/" + sequence_name + '.npy')
    hmm_feature = np.load( "../实验数据2/hhm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, esm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load("../实验数据2/DSSP/" + sequence_name + '_dssp.npy')
    return dssp_feature.astype(np.float32)              

def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def cal_edges(sequence_name, radius=MAP_CUTOFF): 
    dist_matrix = np.load("../实验数据2/distance_matrices/" + sequence_name + '_distance_matrix.npy')
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(np.int)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]

    return radius_index_list

def load_graph(sequence_name):
    dismap = np.load("../实验数据2/distance_matrices/" + sequence_name + '_distance_matrix.npy')
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(np.int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix = map(list, zip(*samples))
    label = torch.Tensor(label)
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix)
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix


class ProDataset(Dataset):

    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM, 
                 psepos_path='./Dataset/protein_dict.pkl'):
        self.names = dataframe['ID'].values 
        self.sequences = dataframe['sequence'].values 
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb')) 
        self.radius = radius 
        self.dist = dist 


    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        nodes_num = len(sequence)
        pos = self.residue_psepos[sequence_name]
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)
        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features], axis=1)

        node_features = torch.from_numpy(node_features)
        node_features= node_features + 0.5 * torch.randn_like(node_features)

        radius_index_list = cal_edges(sequence_name, MAP_CUTOFF)
        edge_feat = self.cal_edge_attr(radius_index_list, pos) 
        G = dgl.DGLGraph()
        G.add_nodes(nodes_num)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)

        self.add_edges_custom(G,
                              radius_index_list,
                              edge_feat
                              )

        adj_matrix = load_graph(sequence_name)
        node_features = node_features.detach().numpy()

        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    
        return sequence_name, sequence, label, node_features, G, adj_matrix

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2,keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)

        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list

    def add_edges_custom(self, G, radius_index_list, edge_features):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)




class DeepHotResi(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, dropout):
        super(DeepHotResi, self).__init__()
        self.liner = nn.Linear(in_features=nhidden*4+nfeat, out_features=nclass)
        self.criterion = FocalLoss(alpha=0.4535, gamma=2.0, reduction='mean') 
        self.gatv2conv = GATConv(nfeat, nhidden, num_heads=4)
        self.dropout = nn.Dropout(dropout)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=10, min_lr=1e-6)
        self.se = SELayer(2614, 16)

    def forward(self, x, graph, adj_matrix):
        x = x.float()
        x = x.view([x.shape[0]*x.shape[1], x.shape[2]])
        x = self.se(x)
        xse = x
        graph = dgl.add_self_loop(graph)
        x = self.gatv2conv(graph, x,)
        x = x.view([x.shape[0], -1])
        x = torch.cat([x, xse],1)
        x = self.dropout(x)
        x = self.liner(x)

        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        l, c = x.size()
        y = torch.mean(x, dim=0)
        y = self.fc(y)

        return x * y


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:  # 'none'
            return F_loss
