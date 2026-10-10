'''
@inproceedings{koren2008factorization,
  title={Factorization meets the neighborhood: a multifaceted collaborative filtering model},
  author={Koren, Yehuda},
  booktitle={Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={426--434},
  year={2008}
}
'''
import torch
import torch.nn as nn
import numpy as np

from daisy.model.AbstractRecommender import GeneralRecommender


class SVDPP(GeneralRecommender):
    tunable_param_names = ['num_ng', 'factors', 'lr', 'batch_size', 'reg_1', 'reg_2', 'reg_y']

    def __init__(self, config):
        """
        SVD++ Recommender Class

        SVD++ extends basic matrix factorization by incorporating implicit feedback.
        It models user preferences using both explicit ratings and implicit feedback
        from items the user has interacted with.

        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factors
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term for user/item embeddings
        reg_2 : float, second-order regularization term for user/item embeddings
        reg_y : float, regularization term for implicit feedback embeddings
        loss_type : str, loss function type
        optimizer : str, optimization method for training
        initializer: str, parameter initializer
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(SVDPP, self).__init__(config)

        self.lr = config['lr']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']
        self.reg_y = config['reg_y']
        self.epochs = config['epochs']
        self.topk = config['topk']

        # User and item embeddings (explicit factors)
        self.embed_user = nn.Embedding(config['user_num'], config['factors'])
        self.embed_item = nn.Embedding(config['item_num'], config['factors'])

        # Implicit feedback embeddings
        # y_j: implicit feedback for items
        self.embed_item_implicit = nn.Embedding(config['item_num'], config['factors'])

        # User and item biases
        self.user_bias = nn.Embedding(config['user_num'], 1)
        self.item_bias = nn.Embedding(config['item_num'], 1)

        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.loss_type = config['loss_type']
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'adam'
        self.initializer = config['init_method'] if config['init_method'] != 'default' else 'normal'
        self.early_stop = config['early_stop']

        self.apply(self._init_weight)

        # Initialize biases to small values
        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.item_bias.weight, 0.0)

    def forward(self, user, item, user_implicit_items=None, user_implicit_counts=None):
        """
        Forward pass for SVD++

        Parameters
        ----------
        user : torch.Tensor, user IDs
        item : torch.Tensor, item IDs
        user_implicit_items : torch.Tensor, items that users have interacted with (for implicit feedback)
        user_implicit_counts : torch.Tensor, number of items each user has interacted with

        Returns
        -------
        pred : torch.Tensor, predicted ratings
        """
        # Get explicit factors
        user_emb = self.embed_user(user)  # batch_size x factors
        item_emb = self.embed_item(item)  # batch_size x factors

        # Add implicit feedback if available
        if user_implicit_items is not None and user_implicit_counts is not None:
            # Get implicit item embeddings and aggregate
            implicit_emb = self.embed_item_implicit(user_implicit_items)  # batch_size x max_items x factors

            # Sum implicit feedback and normalize by sqrt(|R(u)|)
            implicit_sum = implicit_emb.sum(dim=1)  # batch_size x factors

            # Normalize by the square root of the number of items
            user_implicit_counts = user_implicit_counts.unsqueeze(1).float()  # batch_size x 1
            normalized_implicit = implicit_sum / torch.sqrt(user_implicit_counts + 1e-8)

            # Combine explicit and implicit user factors
            user_emb = user_emb + normalized_implicit

        # Get biases
        user_b = self.user_bias(user).squeeze()  # batch_size
        item_b = self.item_bias(item).squeeze()  # batch_size

        # Compute prediction: global_bias + user_bias + item_bias + user_emb · item_emb
        pred = self.global_bias + user_b + item_b + (user_emb * item_emb).sum(dim=-1)

        return pred

    def calc_loss(self, batch):
        """
        Calculate loss for a batch

        For BPR loss: batch = [user, pos_item, neg_item]
        For pointwise loss: batch = [user, item, rating]
        """
        user = batch[0].to(self.device)
        pos_item = batch[1].to(self.device)

        # For simplicity, we don't use implicit feedback in loss calculation
        # In a full implementation, you would pass user interaction history
        pos_pred = self.forward(user, pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            # Pointwise loss (classification or regression)
            label = batch[2].to(self.device).float()
            loss = self.criterion(pos_pred, label)

            # Add regularization
            loss += self.reg_1 * (self.embed_user(user).norm(p=1) + self.embed_item(pos_item).norm(p=1))
            loss += self.reg_2 * (self.embed_user(user).norm() + self.embed_item(pos_item).norm())
            loss += self.reg_y * self.embed_item_implicit(pos_item).norm()

        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            # Pairwise loss
            neg_item = batch[2].to(self.device)
            neg_pred = self.forward(user, neg_item)
            loss = self.criterion(pos_pred, neg_pred)

            # Add regularization
            loss += self.reg_1 * (self.embed_user(user).norm(p=1) +
                                 self.embed_item(pos_item).norm(p=1) +
                                 self.embed_item(neg_item).norm(p=1))
            loss += self.reg_2 * (self.embed_user(user).norm() +
                                 self.embed_item(pos_item).norm() +
                                 self.embed_item(neg_item).norm())
            loss += self.reg_y * (self.embed_item_implicit(pos_item).norm() +
                                 self.embed_item_implicit(neg_item).norm())
        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        return loss

    def predict(self, u, i):
        """
        Predict rating for a single user-item pair
        """
        u = torch.tensor(u, device=self.device)
        i = torch.tensor(i, device=self.device)
        pred = self.forward(u, i).cpu().item()

        return pred

    def rank(self, test_loader):
        """
        Rank items for users in the test set

        Returns
        -------
        rec_ids : numpy.ndarray, top-k recommended item IDs for each user
        """
        rec_ids = torch.tensor([], device=self.device)

        for us, cands_ids in test_loader:
            us = us.to(self.device)
            cands_ids = cands_ids.to(self.device)

            batch_size = us.shape[0]
            cand_num = cands_ids.shape[1]

            # Expand user embeddings to match candidate items
            user_emb = self.embed_user(us).unsqueeze(dim=1)  # batch x 1 x factors
            user_emb = user_emb.expand(batch_size, cand_num, -1)  # batch x cand_num x factors

            # Get item embeddings
            item_emb = self.embed_item(cands_ids)  # batch x cand_num x factors

            # Get biases
            user_b = self.user_bias(us)  # batch x 1
            item_b = self.item_bias(cands_ids).squeeze(dim=-1)  # batch x cand_num

            # Compute scores: global_bias + user_bias + item_bias + user_emb · item_emb
            scores = self.global_bias + user_b + item_b + (user_emb * item_emb).sum(dim=-1)

            # Rank items by scores
            rank_ids = torch.argsort(scores, descending=True)
            rank_list = torch.gather(cands_ids, 1, rank_ids)[:, :self.topk]

            rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy()

    def full_rank(self, u):
        """
        Rank all items for a single user

        Parameters
        ----------
        u : int, user ID

        Returns
        -------
        top_items : numpy.ndarray, top-k item IDs
        """
        u = torch.tensor(u, device=self.device)

        user_emb = self.embed_user(u)  # factors
        items_emb = self.embed_item.weight  # item_num x factors

        # Get biases
        user_b = self.user_bias(u)  # 1
        items_b = self.item_bias.weight.squeeze()  # item_num

        # Compute scores for all items
        scores = self.global_bias + user_b + items_b + torch.matmul(user_emb, items_emb.T)

        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()
