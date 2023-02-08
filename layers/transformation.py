import torch
import numpy as np
import scipy.sparse as sp


class TransformationPOS:
    """
    Graph augmentation: randomly mask out edges and features
    """
    def __init__(self, p_drop):
        self.drop = torch.nn.Dropout(p_drop)

    def __call__(self, feature, adj):
        feature = self.drop(feature)
        adj = self.drop(adj)
        return feature, adj


class TransformationNEG:
    """
    Random shuffling.
    """
    def __call__(self, feature, adj):
        nb_nodes = feature.shape[0]
        idx = np.arange(0, nb_nodes)
        positions = np.random.uniform(0, 1, nb_nodes) < 1
        idx_shuffle = np.nonzero(positions)[0]
        np.random.shuffle(idx_shuffle)
        idx[positions] = idx_shuffle
        shuf_fts = feature[idx, :].to(feature.device)
        return shuf_fts, adj


class Transformation:
    def __init__(self, p_feat, p_adj, sc):
        self.p_feat = p_feat
        self.p_adj = p_adj
        self.sc = sc   # self-connection

    def __call__(self, feature, adj):
        feature = self.random_mask(feature, p=self.p_feat)
        adj = self.random_mask(adj, p=self.p_adj)

        feature = self.process_feat(feature)
        adj = self.process_adj(adj)
        return feature, adj

    def random_mask(self, x, p):
        idx = np.random.uniform(0, 1, x.shape)
        idx = idx > p
        x *= idx
        return x

    def process_feat(self, feature):
        feature = torch.FloatTensor(feature)
        return feature

    def process_adj(self, adj):
        adj += np.eye(adj.shape[0])*self.sc
        adj = sp.csr_matrix(adj)
        adj = self.normalize_adj(adj)
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
