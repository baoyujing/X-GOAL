import os
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .model import Model
from .encoder import Encoder

from evaluate import evaluate


class XGOAL(Model):
    def __init__(self, args):
        super().__init__(args)
        self.encoder = Encoder(ft_size=args.ft_size, hid_units=args.hid_units, n_adj=len(self.adj_list)).to(args.device)

    def get_embeddings(self):
        self.encoder.eval()
        final_embs = []
        for n, adj in enumerate(self.adj_list):
            embeds = self.encoder(self.features, adj, n)
            embeds = embeds.detach()
            final_embs.append(embeds)
        final_embs = torch.mean(torch.stack(final_embs), 0)   # average pooling
        return final_embs

    def train(self):
        print("Started training on {} with {}...".format(self.args.dataset, self.args.model))
        if self.args.is_warmup:
            self.opt = torch.optim.Adam(self.encoder.parameters(), lr=self.args.warmup_lr)
            self.warmup()
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=self.args.lr)
        if self.args.pretrained_model_path:
            path = self.args.pretrained_model_path
        else:
            path = os.path.join(self.args.save_root, 'warmup_{}_{}.pkl'.format(self.args.dataset, self.args.model))
        self.encoder.load_state_dict(torch.load(path))
        self._train_full_loss()

    def evaluate(self, path=""):
        if path:
            print("Evaluating based on {}".format(path))
            self.encoder.load_state_dict(torch.load(path))
        embs = self.get_embeddings()
        macro_f1s, micro_f1s, nmi, sim = evaluate(embs, self.idx_train, self.idx_val, self.idx_test, self.labels)
        return macro_f1s, micro_f1s, nmi, sim

    def _train_full_loss(self):
        print("Full loss training...")
        cnt_wait = 0
        best = 1e9
        self.encoder.train()

        idx2center = defaultdict(torch.Tensor)
        idx2target = defaultdict(torch.Tensor)
        for n_epoch in tqdm(range(self.args.nb_epochs)):
            self.opt.zero_grad()

            # nce loss
            loss_nce_list = []
            h_list = []
            h_list_pos = []
            h_list_neg = []
            w_list = self.args.w_list
            for n, adj in enumerate(self.adj_list):
                feature_pos, adj_pos = self.transform_pos(self.features, adj)
                feature_neg, adj_neg = self.transform_neg(self.features, adj)

                h_org = self.encoder(self.features, adj, n)
                h_pos = self.encoder(feature_pos, adj_pos, n)
                h_neg = self.encoder(feature_neg, adj_neg, n)
                h_list.append(h_org)
                h_list_pos.append(h_pos)
                h_list_neg.append(h_neg)

                loss_nce = self.info_nce(h_org=h_org, h_pos=h_pos, h_neg=h_neg)
                loss_nce *= w_list[n]
                loss_nce_list.append(loss_nce)
            loss_reg_n = self.node_alignment(h_list, h_list_neg)
            loss_reg_n *= self.args.w_reg_n

            # clustering
            if n_epoch % self.args.cluster_step == 0:
                for n in range(self.n_adj):
                    centroids, target = self.run_kmeans(x=h_list[n], k=self.args.k[n])
                    idx2center[n] = centroids
                    idx2target[n] = target

            loss_cluster_list = []
            for n in range(self.n_adj):
                loss_cluster = self.loss_kmeans(x=h_list[n], centroids=idx2center[n], tau=self.args.tau[n], target=idx2target[n])
                loss_cluster *= w_list[n]
                loss_cluster_list.append(loss_cluster)

            loss_nce, loss_cluster = torch.sum(torch.stack(loss_nce_list)), torch.sum(torch.stack(loss_cluster_list))
            loss_cluster *= self.args.w_cluster

            # regularization
            center_list = [idx2center[n] for n in range(self.n_adj)]
            loss_reg_c = self.regularize_k(h_list=h_list, center_list=center_list)

            loss = loss_nce + loss_cluster + loss_reg_n + self.args.w_reg_c*loss_reg_c

            if n_epoch % 10 == 0:
                print("loss_full: {:.6f}, L_n: {:.6}, L_c: {:.6}, R_n: {:.6}, R_c: {:.6}".format(
                    loss.detach().cpu().numpy(), loss_nce.detach().cpu().numpy(), loss_cluster.detach().cpu().numpy(),
                    loss_reg_n.detach().cpu().numpy(), loss_reg_c.detach().cpu().numpy()))

            # early stop
            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                torch.save(self.encoder.state_dict(), os.path.join(
                    self.args.save_root, '{}_{}.pkl'.format(self.args.dataset, self.args.model)))
                break

            loss.backward()
            self.opt.step()

    def regularize_k(self, h_list, center_list):
        loss_list = []
        for n in range(self.n_adj):
            # for a set of clusters, obtain the codes
            center = center_list[n]
            p_list, log_p_list = [], []
            for n_i, h in enumerate(h_list):
                code = torch.matmul(h, center)   # [n_nodes, k]
                p = F.softmax(code, dim=-1)
                log_p = torch.log(p + 1e-12)
                p_list.append(p)
                log_p_list.append(log_p)    # [n_adj, n_nodes, k]

            # compare the codes
            # KL divergence
            kl_list = []
            for i in range(self.n_adj):
                if n == i:
                    continue
                p = p_list[n]
                p = p.detach()
                kl = p*torch.log(p/p_list[i])
                kl = torch.sum(kl, dim=1)
                kl_list.append(torch.mean(kl))
            kl = torch.mean(torch.stack(kl_list))
            loss_list.append(kl)
        loss = torch.sum(torch.stack(loss_list))
        return loss

    def warmup(self):
        print("Warming up...")
        cnt_wait = 0
        best = 1e9
        self.encoder.train()
        for n_epoch in tqdm(range(self.args.nb_epochs)):
            self.opt.zero_grad()
            loss_nce_list = []
            h_list = []
            h_list_pos = []
            h_list_neg = []
            w_list = self.args.w_list
            for n, adj in enumerate(self.adj_list):
                feature_pos, adj_pos = self.transform_pos(self.features, adj)
                feature_neg, adj_neg = self.transform_neg(self.features, adj)

                h_org = self.encoder(self.features, adj, n)
                h_pos = self.encoder(feature_pos, adj_pos, n)
                h_neg = self.encoder(feature_neg, adj_neg, n)
                h_list.append(h_org)
                h_list_pos.append(h_pos)
                h_list_neg.append(h_neg)

                loss_nce = self.info_nce(h_org=h_org, h_pos=h_pos, h_neg=h_neg)
                loss_nce *= w_list[n]
                loss_nce_list.append(loss_nce)

            loss_reg_n = self.node_alignment(h_list, h_list_neg)
            loss_reg_n *= self.args.warmup_w_reg_n

            loss_nce = torch.sum(torch.stack(loss_nce_list))
            loss = loss_nce + loss_reg_n

            if n_epoch % 100 == 0:
                print("Loss_full: {:.6f}, L_n: {:.6f}, R_n: {:.6f}".format(
                    loss.detach().cpu().numpy(), loss_nce.detach().cpu().numpy(), loss_reg_n.detach().cpu().numpy()))

            # early stop
            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(self.encoder.state_dict(), os.path.join(self.args.save_root, 'warmup_{}_{}_test.pkl'.format(
                    self.args.dataset, self.args.model)))
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                break

            loss.backward()
            self.opt.step()
