import os
from tqdm import tqdm


import torch

from .model import Model
from .encoder import Encoder

from evaluate import evaluate


class GOAL(Model):
    def __init__(self, args):
        super().__init__(args)
        self.adj = self.adj_list[self.args.layer].to(self.args.device)
        self.encoder = Encoder(self.args.ft_size, self.args.hid_units).to(self.args.device)

    def get_embeddings(self):
        self.encoder.eval()
        embeds = self.encoder(self.features, self.adj)
        embeds = embeds.detach()
        return embeds

    def train(self):
        print("Started training on {}-{} layer with {}...".format(self.args.dataset, self.args.layer, self.args.model))
        if self.args.is_warmup:
            self.opt = torch.optim.Adam(self.encoder.parameters(), lr=self.args.warmup_lr)
            self.warmup()
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=self.args.lr)
        if self.args.pretrained_model_path:
            path = self.args.pretrained_model_path
        else:
            path = os.path.join(self.args.save_root, 'warmup_{}_{}_{}.pkl'.format(
                self.args.dataset, self.args.model, self.args.layer))
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
        print("Full training loss...")
        cnt_wait = 0
        best = 1e9
        self.encoder.train()
        for n_epoch in tqdm(range(self.args.nb_epochs)):
            self.opt.zero_grad()

            feature_pos, adj_pos = self.transform_pos(self.features, self.adj)
            feature_neg, adj_neg = self.transform_neg(self.features, self.adj)

            h_org = self.encoder(self.features, self.adj)
            h_pos = self.encoder(feature_pos, adj_pos)
            h_neg = self.encoder(feature_neg, adj_neg)

            loss_nce = self.info_nce(h_org=h_org, h_pos=h_pos, h_neg=h_neg)
            if n_epoch % self.args.cluster_step == 0:
                centroids, target = self.run_kmeans(x=h_org, k=self.args.k)
            loss_cluster = self.loss_kmeans(x=h_org, centroids=centroids, tau=self.args.tau, target=target)
            loss = loss_nce + self.args.w_cluster*loss_cluster
            if n_epoch % 10 == 0:
                print(loss, loss_nce, loss_cluster)

            # early stop
            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                torch.save(self.encoder.state_dict(), os.path.join(self.args.save_root, '{}_{}_{}.pkl'.format(
                    self.args.dataset, self.args.model, self.args.layer)))
                break

            loss.backward()
            self.opt.step()

    def warmup(self):
        print("Warming up...")
        cnt_wait = 0
        best = 1e9
        self.encoder.train()
        for n_epoch in tqdm(range(self.args.nb_epochs)):
            self.opt.zero_grad()

            feature_pos, adj_pos = self.transform_pos(self.features, self.adj)
            feature_neg, adj_neg = self.transform_neg(self.features, self.adj)

            h_org = self.encoder(self.features, self.adj)
            h_pos = self.encoder(feature_pos, adj_pos)
            h_neg = self.encoder(feature_neg, adj_neg)

            loss = self.info_nce(h_org=h_org, h_pos=h_pos, h_neg=h_neg)
            if n_epoch % 100 == 0:
                print("L_n: {:.6f}".format(loss.detach().cpu().numpy()))

            # early stop
            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                torch.save(self.encoder.state_dict(), os.path.join(self.args.save_root, 'warmup_{}_{}_{}.pkl'.format(
                    self.args.dataset, self.args.model, self.args.layer)))
                break

            loss.backward()
            self.opt.step()
