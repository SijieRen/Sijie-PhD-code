"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder for Unsupervised Clustering

"""
from cProfile import label
from cmath import pi
from email.mime import image
from turtle import numinput
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from networks.Networks import *
from losses.LossFunctions import *
from metrics.Metrics import *
import matplotlib.pyplot as plt


class GMVAE:

    def __init__(self, args):
        self.args = args
        self.num_epochs = args.epochs
        self.cuda = args.cuda
        self.verbose = args.verbose

        self.batch_size = args.batch_size
        self.batch_size_val = args.batch_size_val
        self.learning_rate = args.learning_rate
        self.decay_epoch = args.decay_epoch
        self.lr_decay = args.lr_decay
        self.w_cat = args.w_categ
        self.w_gauss = args.w_gauss
        self.w_rec = args.w_rec
        self.rec_type = args.rec_type
        self.w_balance1 = args.w_balance1
        self.w_balance2 = args.w_balance2
        self.if_balance = args.if_balance

        self.num_classes = args.num_classes
        self.gaussian_size = args.gaussian_size
        self.input_size = args.input_size

        # gumbel
        self.init_temp = args.init_temp
        self.decay_temp = args.decay_temp
        self.hard_gumbel = args.hard_gumbel
        self.min_temp = args.min_temp
        self.decay_temp_rate = args.decay_temp_rate
        self.gumbel_temp = self.init_temp

        self.network = GMVAENet(
            self.input_size, self.gaussian_size, self.num_classes)
        self.network_big = GMVAENet_big(
            self.input_size, self.gaussian_size, self.num_classes)
        self.model_MI = MINE(data_dim=1, hidden_size=10).cuda()
        # model_MI = MINE(data_dim=1, hidden_size=10).cuda()
        self.optimizer_MI = torch.optim.Adam(
            self.model_MI.parameters(), lr=0.002)
        self.losses = LossFunctions()
        self.metrics = Metrics()

        if self.cuda:
            self.network = self.network.cuda()
            self.network_big = self.network_big.cuda()

    def unlabeled_loss(self, data, out_net, labels, model_MI, optimizer_MI):
        """Method defining the loss functions derived from the variational lower bound
        Args:
            data: (array) corresponding array containing the input data
            out_net: (dict) contains the graph operations or nodes of the network output

        Returns:
            loss_dic: (dict) contains the values of each loss function and predictions
        """
        # obtain network variables
        z, data_recon = out_net['gaussian'], out_net['x_rec']
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']
        predict_e = out_net["categorical"]

        # reconstruction loss
        loss_rec = self.losses.reconstruction_loss(
            data, data_recon, self.rec_type)

        # gaussian loss
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        # print("logits: ", logits)
        # print("prob_cat: ", prob_cat)
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        # balance and mutual loss
        like_tensor = torch.full_like(logits, 1.0/self.num_classes)
        _, predicted_labels = torch.max(logits, dim=1)
        # print("predict", predicted_labels.size())
        # print("target", labels.size())
        # loss_balance = self.w_balance*(self.losses.mutual_loss(predict_e, labels) +
        #                                self.losses.entropy(like_tensor, prob_cat))

        # for epoch in range(10):
        #     optimizer_MI.zero_grad()
        #     loss_MI = model_MI(predicted_labels/(self.num_classes-1), labels)

        #     loss_MI.backward()  # retain_graph=True
        #     optimizer_MI.step()
        # # all_mi.append(-loss.item())

        # minist 2   NICO 2   ad3  SUOYONG LOSS
        # if 'mnist' in self.args.dataset or 'NICO' in self.args.dataset:
        #     loss_MI, _, _ = mine_2(prob_cat, labels)
        # elif 'AD' in self.args.dataset:
        #     loss_MI, _, _ = mine_M(prob_cat, labels)

        MI_tensor = F.one_hot(labels, num_classes=torch.max(labels)+1).cuda()
        k1 = self.num_classes
        k2 = MI_tensor.size(1)
        p_ij = prob_cat.unsqueeze(2) * MI_tensor.unsqueeze(1)
        # p_ij_p = like_tensor.unsqueeze(2) * MI_tensor.unsqueeze(1) + 1e-4
        p_ij = p_ij.sum(dim=0)
        p_ij /= p_ij.sum()
        p_i = (p_ij.sum(dim=1).view(k1, 1).expand(k1, k2))
        p_j = p_ij.sum(dim=0).view(1, k2).expand(k1, k2)
        # print("pi: ", p_i)
        # print("pj: ", p_j)
        # print("pij: ", p_ij)
        loss_MI = p_ij * (torch.log(p_ij + 1e-10) -
                          torch.log(p_j + 1e-10) - torch.log(p_i + 1e-10))
        loss_MI = loss_MI.sum()
        # MI_tensor = F.one_hot(labels, num_classes=self.num_classes)
        # loss_MI = F.kl_div(p_ij_a.softmax(
        #     dim=-1).log(), p_ij_p.softmax(dim=-1).log())

        # print(loss_MI)
        # loss_balance = torch.sum(torch.pow(prob_cat, 2))
        #loss_balance = F.mse_loss(like_tensor, prob_cat)
        loss_balance = F.kl_div(prob_cat.softmax(
            dim=-1).log(), like_tensor.softmax(dim=-1).log())
        loss_MI_balance = self.w_balance1*loss_MI + \
            self.w_balance2*loss_balance
        # balance loss
        # if self.if_balance:
        #     like_tensor = torch.full_like(logits, 1.0/self.num_classes)
        #     loss_balance = self.w_balance * \
        #         self.losses.entropy(like_tensor, prob_cat)

        # total loss
        loss_total = self.w_rec * loss_rec + \
            self.w_gauss * loss_gauss + self.w_cat * loss_cat + loss_MI_balance

        # obtain predictions
        _, predicted_labels = torch.max(logits, dim=1)
        # print(loss_total)
        loss_dic = {'total': loss_total,
                    'predicted_labels': predicted_labels,
                    'reconstruction': loss_rec,
                    'gaussian': loss_gauss,
                    'mutual': loss_MI,
                    'categorical': loss_cat,
                    'balance': loss_balance, }
        return loss_dic

    def train_epoch(self, optimizer, data_loader):
        """Train the model for one epoch

        Args:
            optimizer: (Optim) optimizer to use in backpropagation
            data_loader: (DataLoader) corresponding loader containing the training data

        Returns:
            average of all loss values, accuracy, nmi
        """
        self.network.train()
        total_loss = 0.
        recon_loss = 0.
        cat_loss = 0.
        gauss_loss = 0.
        mutual_loss = 0.
        balance_loss = 0.
        eps = 1e-5

        accuracy = 0.
        nmi = 0.
        num_batches = 0.
        num_batch_env = torch.zeros(
            [self.args.num_classes, 1]).cuda()
        true_labels_list = []
        predicted_labels_list = []
        balance_0_tensor = torch.zeros(
            [self.args.num_classes, 1]).requires_grad_(True).cuda()
        balance_1_tensor = torch.zeros(
            [self.args.num_classes, 1]).requires_grad_(True).cuda()
        balance_0_tensor_sum = torch.zeros(
            [self.args.num_classes, 1]).requires_grad_(True).cuda()
        balance_1_tensor_sum = torch.zeros(
            [self.args.num_classes, 1]).requires_grad_(True).cuda()
        balance_like_tensor_mse = torch.zeros(
            [self.args.num_classes, 1]).cuda()

        # print(balance_like_tensor.size())
        # iterate over the dataset
        if 'mnist' in self.args.dataset or 'NICO' in self.args.dataset:
            for batch_idx, (data, labels, env) in enumerate(data_loader):

                # if data.size(0) != self.args.batch_size:
                #     continue
                # else:
                if self.cuda == 1:
                    data = data.cuda()
                    labels = labels.cuda()
                # print("data", data.cpu().numpy()[1].tolist())
                optimizer.zero_grad()

                # flatten data
                data = data.view(data.size(0), -1)

                # forward call
                out_net = self.network(
                    data, self.gumbel_temp, self.hard_gumbel)

                unlab_loss_dic = self.unlabeled_loss(
                    data, out_net, labels, self.model_MI, self.optimizer_MI)
                total = unlab_loss_dic['total']

                # accumulate values
                total_loss += total.item()
                recon_loss += unlab_loss_dic['reconstruction'].item()
                gauss_loss += unlab_loss_dic['gaussian'].item()
                cat_loss += unlab_loss_dic['categorical'].item()
                mutual_loss += unlab_loss_dic['mutual'].item()
                balance_loss += unlab_loss_dic['balance'].item()

                # save predicted and true labels
                predicted = unlab_loss_dic['predicted_labels']
                true_labels_list.append(labels)
                predicted_labels_list.append(predicted)

                p_e = 1./self.args.num_classes

                total.backward()
                optimizer.step()

                num_batches += 1.

        elif 'AD' in self.args.dataset:
            for batch_idx, (data, u, us, labels, env) in enumerate(data_loader):
                # if data.size(0) != self.args.batch_size:
                #     continue
                # else:
                # print(batch_idx, data.size(0))
                if self.cuda == 1:
                    data = data.cuda()
                    labels = labels.cuda()
                # print("data", data.cpu().numpy()[1].tolist())
                optimizer.zero_grad()

                # flatten data
                data = data.view(data.size(0), -1)

                # forward call
                out_net = self.network(
                    data, self.gumbel_temp, self.hard_gumbel)
                unlab_loss_dic = self.unlabeled_loss(
                    data, out_net, labels, self.model_MI, self.optimizer_MI)
                total = unlab_loss_dic['total']

                # accumulate values
                total_loss += total.item()
                recon_loss += unlab_loss_dic['reconstruction'].item()
                gauss_loss += unlab_loss_dic['gaussian'].item()
                cat_loss += unlab_loss_dic['categorical'].item()
                mutual_loss += unlab_loss_dic['mutual'].item()
                balance_loss += unlab_loss_dic['balance'].item()

                # perform backpropagation
                total.backward()
                optimizer.step()

                # save predicted and true labels
                predicted = unlab_loss_dic['predicted_labels']
                true_labels_list.append(labels)
                predicted_labels_list.append(predicted)

                p_e = 1./self.args.num_classes

                num_batches += 1.

        # average per batch
        total_loss /= num_batches
        recon_loss /= num_batches
        gauss_loss /= num_batches
        cat_loss /= num_batches
        mutual_loss /= num_batches
        balance_loss /= num_batches
        # balance_1_tensor /= (num_batch_env + eps)
        # balance_0_tensor /= (num_batch_env + eps)
        # print("Mutual info loss: ", mutual_loss_sum)

        # # concat all true and predicted labels
        # true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
        # predicted_labels = torch.cat(
        #     predicted_labels_list, dim=0).cpu().numpy()

        # balance_like_tensor /= num_batches
        # print(balance_like_tensor)
        # global balence in each env backpropagation\
        # if self.if_balance:
        #     # balance_loss = self.args.w_balance * \
        #     #     torch.add(balance_loss, torch.mean(
        #     #         balance_like_tensor))

        #     balance_loss = self.args.w_balance1 * \
        #         F.mse_loss(balance_1_tensor_sum,
        #                    balance_0_tensor_sum, reduction="mean")
        #     optimizer.zero_grad()
        #     balance_loss.backward()
        #     optimizer.step()

        # compute metrics
        # accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
        # nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)

        # , accuracy, nmi
        return total_loss, recon_loss, gauss_loss, cat_loss, mutual_loss, balance_loss

    def test(self, data_loader, return_loss=False):
        """Test the model with new data

        Args:
            data_loader: (DataLoader) corresponding loader containing the test/validation data
            return_loss: (boolean) whether to return the average loss values

        Return:
            accuracy and nmi for the given test data

        """
        self.network.eval()
        total_loss = 0.
        recon_loss = 0.
        cat_loss = 0.
        gauss_loss = 0.

        accuracy = 0.
        nmi = 0.
        num_batches = 0

        true_labels_list = []
        predicted_labels_list = []
        image_path_list = []
        image_path_num_start = 0

        with torch.no_grad():
            if 'mnist' in self.args.dataset or 'NICO' in self.args.dataset:
                for batch_idx, (data, labels, env) in enumerate(data_loader):
                    # print(data.size())
                    # if data.size(0) != self.args.batch_size:
                    #     continue
                    # else:
                    if self.cuda == 1:
                        data = data.cuda()
                        labels = labels.cuda()

                    # optimizer.zero_grad()

                    # flatten data
                    data = data.view(data.size(0), -1)

                    # forward call
                    if 'mnist' in self.args.dataset:
                        out_net = self.network(
                            data, self.gumbel_temp, self.hard_gumbel)
                    elif 'NICO' in self.args.dataset:
                        out_net = self.network_big(
                            data, self.gumbel_temp, self.hard_gumbel)
                    unlab_loss_dic = self.unlabeled_loss(
                        data, out_net, labels, self.model_MI, self.optimizer_MI)
                    total = unlab_loss_dic['total']

                    # accumulate values
                    total_loss += total.item()
                    recon_loss += unlab_loss_dic['reconstruction'].item()
                    gauss_loss += unlab_loss_dic['gaussian'].item()
                    cat_loss += unlab_loss_dic['categorical'].item()

                    # perform backpropagation
                    # total.backward()
                    # optimizer.step()

                    # save predicted and true labels
                    predicted = unlab_loss_dic['predicted_labels']
                    true_labels_list.append(labels)
                    predicted_labels_list.append(predicted)
                    # image_path_num_start = num_batches*data.size(0)
                    image_path_num_end = image_path_num_start + data.size(0)
                    for jj in range(image_path_num_start, image_path_num_end):
                        image_path_list.append(
                            data_loader.dataset.image_path_list[jj])  # 有点蠢了
                        # true_labels_list.append(labels.[jj])
                    image_path_num_start = image_path_num_end

                    num_batches += 1
            elif 'AD' in self.args.dataset:
                for batch_idx, (data, u, us, labels, env) in enumerate(data_loader):
                    # if data.size(0) != self.args.batch_size:
                    #     continue
                    # else:
                    if self.cuda == 1:
                        data = data.cuda()
                        labels = labels.cuda()

                    # optimizer.zero_grad()

                    # flatten data
                    data = data.view(data.size(0), -1)

                    # forward call
                    out_net = self.network_big(
                        data, self.gumbel_temp, self.hard_gumbel)
                    unlab_loss_dic = self.unlabeled_loss(
                        data, out_net, labels, self.model_MI, self.optimizer_MI)
                    total = unlab_loss_dic['total']

                    # accumulate values
                    total_loss += total.item()
                    recon_loss += unlab_loss_dic['reconstruction'].item()
                    gauss_loss += unlab_loss_dic['gaussian'].item()
                    cat_loss += unlab_loss_dic['categorical'].item()

                    # perform backpropagation
                    # total.backward()
                    # optimizer.step()

                    # save predicted and true labels
                    predicted = unlab_loss_dic['predicted_labels']
                    true_labels_list.append(labels)
                    predicted_labels_list.append(predicted)

                    image_path_num_end = image_path_num_start + data.size(0)
                    # print("start: ", image_path_num_start)
                    # print("stop: ", image_path_num_end)
                    for jj in range(image_path_num_start, image_path_num_end):
                        image_path_list.append(
                            data_loader.dataset.image_path_list[jj])
                    image_path_num_start = image_path_num_end
                    num_batches += 1

        # average per batch
        # if return_loss:
        total_loss /= num_batches
        recon_loss /= num_batches
        gauss_loss /= num_batches
        cat_loss /= num_batches

        # concat all true and predicted labels
        true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
        print(torch.cat(
            predicted_labels_list, dim=0).size(0))
        predicted_labels = torch.cat(
            predicted_labels_list, dim=0).cpu().numpy()

        # image_paths = torch.cat(image_path_list, dim=0).cpu().numpy()
        # print(predicted_labels)

        return {'total_loss': total_loss, 'rec_loss': recon_loss,
                'gauss_loss': gauss_loss, 'cat_loss': cat_loss,
                'predict_label': predicted_labels.tolist(), 'image_path': image_path_list,
                'true_label': true_labels.tolist()}

    def train(self, train_loader, val_loader):
        """Train the model

        Args:
            train_loader: (DataLoader) corresponding loader containing the training data
            val_loader: (DataLoader) corresponding loader containing the validation data

        Returns:
            output: (dict) contains the history of train/val loss
        """
        if 'mnist' in self.args.dataset:
            optimizer = optim.Adam(self.network.parameters(),
                                   lr=self.learning_rate)
        elif 'AD' in self.args.dataset or 'NICO' in self.args.dataset:
            optimizer = optim.Adam(self.network_big.parameters(),
                                   lr=self.learning_rate)
        train_history_acc, val_history_acc = [], []
        train_history_nmi, val_history_nmi = [], []

        for epoch in range(1, self.num_epochs + 1):
            # train_loss, train_rec, train_gauss, train_cat, train_acc, train_nmi = self.train_epoch(optimizer, train_loader)
            train_loss, train_rec, train_gauss, train_cat, train_mutual, train_balance = self.train_epoch(
                optimizer, train_loader)
            # val_loss, val_rec, val_gauss, val_cat, val_acc, val_nmi = self.test(val_loader, True)

            # if verbose then print specific information about training
            if self.verbose == 1:
                print("(Epoch %d / %d)" % (epoch, self.num_epochs))
                print("Train - REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;  Mutual: %.5lf, Balance: %.5lf" %
                      (train_rec, train_gauss, train_cat, train_mutual, train_balance))
                # print("Valid - REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
                #       (val_rec, val_gauss, val_cat))
                print("Total Loss=Train: %.5lf" % (train_loss))
            else:
                print('(Epoch %d / %d) Train_Loss: %.3lf' %
                      (epoch, self.num_epochs, train_loss))

            # decay gumbel temperature
            if self.decay_temp == 1:
                self.gumbel_temp = np.maximum(
                    self.init_temp * np.exp(-self.decay_temp_rate * epoch), self.min_temp)
                if self.verbose == 1:
                    print("Gumbel Temperature: %.3lf" % self.gumbel_temp)

            # train_history_acc.append(train_acc)
            # val_history_acc.append(val_acc)
            # train_history_nmi.append(train_nmi)
            # val_history_nmi.append(val_nmi)
        return {'total_loss': train_loss, 'rec_loss': train_rec,
                'gauss_loss': train_gauss, 'cat_loss': train_cat}

    def latent_features(self, data_loader, return_labels=False):
        """Obtain latent features learnt by the model

        Args:
            data_loader: (DataLoader) loader containing the data
            return_labels: (boolean) whether to return true labels or not

        Returns:
           features: (array) array containing the features from the data
        """
        self.network.eval()
        N = len(data_loader.dataset)
        features = np.zeros((N, self.gaussian_size))
        if return_labels:
            true_labels = np.zeros(N, dtype=np.int64)
        start_ind = 0
        with torch.no_grad():
            for (data, labels) in data_loader:
                if self.cuda == 1:
                    data = data.cuda()
                # flatten data
                data = data.view(data.size(0), -1)
                out = self.network.inference(
                    data, self.gumbel_temp, self.hard_gumbel)
                latent_feat = out['mean']
                end_ind = min(start_ind + data.size(0), N+1)

                # return true labels
                if return_labels:
                    true_labels[start_ind:end_ind] = labels.cpu().numpy()
                features[start_ind:end_ind] = latent_feat.cpu().detach().numpy()
                start_ind += data.size(0)
        if return_labels:
            return features, true_labels
        return features

    def reconstruct_data(self, data_loader, sample_size=-1):
        """Reconstruct Data

        Args:
            data_loader: (DataLoader) loader containing the data
            sample_size: (int) size of random data to consider from data_loader

        Returns:
            reconstructed: (array) array containing the reconstructed data
        """
        self.network.eval()

        # sample random data from loader
        indices = np.random.randint(
            0, len(data_loader.dataset), size=sample_size)
        test_random_loader = torch.utils.data.DataLoader(
            data_loader.dataset, batch_size=sample_size, sampler=SubsetRandomSampler(indices))

        # obtain values
        it = iter(test_random_loader)
        test_batch_data, _ = it.next()
        original = test_batch_data.data.numpy()
        if self.cuda:
            test_batch_data = test_batch_data.cuda()

        # obtain reconstructed data
        out = self.network(test_batch_data, self.gumbel_temp, self.hard_gumbel)
        reconstructed = out['x_rec']
        return original, reconstructed.data.cpu().numpy()

    def plot_latent_space(self, data_loader, save=False):
        """Plot the latent space learnt by the model

        Args:
            data: (array) corresponding array containing the data
            labels: (array) corresponding array containing the labels
            save: (bool) whether to save the latent space plot

        Returns:
            fig: (figure) plot of the latent space
        """
        # obtain the latent features
        features = self.latent_features(data_loader)

        # plot only the first 2 dimensions
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(features[:, 0], features[:, 1], c=labels, marker='o',
                    edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
        plt.colorbar()
        if(save):
            fig.savefig('latent_space.png')
        return fig

    def random_generation(self, num_elements=1):
        """Random generation for each category

        Args:
            num_elements: (int) number of elements to generate

        Returns:
            generated data according to num_elements
        """
        # categories for each element
        arr = np.array([])
        for i in range(self.num_classes):
            arr = np.hstack([arr, np.ones(num_elements) * i])
        indices = arr.astype(int).tolist()

        categorical = F.one_hot(torch.tensor(
            indices), self.num_classes).float()

        if self.cuda:
            categorical = categorical.cuda()

        # infer the gaussian distribution according to the category
        mean, var = self.network.generative.pzy(categorical)

        # gaussian random sample by using the mean and variance
        noise = torch.randn_like(var)
        std = torch.sqrt(var)
        gaussian = mean + noise * std

        # generate new samples with the given gaussian
        generated = self.network.generative.pxz(gaussian)

        return generated.cpu().detach().numpy()
