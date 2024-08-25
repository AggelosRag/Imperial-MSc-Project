from sklearn.tree import DecisionTreeClassifier
from torch import nn
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from model.loss import BCEWithLogitsLoss


class ParabolaArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):
        self.model = TreeNet(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config["model"]['lr'],
                                          weight_decay=config["model"]['weight_decay'])
        self.optimizer_sr = torch.optim.Adam(self.model.sr_model.parameters(),
                                             lr=1e-3)
        self.optimizer_mn = torch.optim.Adam(self.model.mn_model.parameters(),
                                             lr=1e-2)
        self.criterion_label = BCEWithLogitsLoss()
        self.criterion_sr = torch.nn.MSELoss()
        self.lr_scheduler = None

class SurrogateNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.feed_forward(x)

class TreeNet(nn.Module):
    def __init__(self, config):
        super(TreeNet, self).__init__()

        self.input_dim = config["dataset"]["num_features"]
        self.output_dim = config["dataset"]["num_classes"]
        self.train_size = 200

        if self.output_dim == 2:
            network_out = 1
            surrogate_input_dim = self.train_size
        else:
            network_out = self.output_dim
            surrogate_input_dim = self.train_size * self.output_dim

        self.mn_model = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 10),
            nn.Tanh(),
            nn.Linear(10, network_out),
        )

        self.sr_model = SurrogateNetwork(surrogate_input_dim)
        self.random_seeds = np.random.randint(1, 100, 10)

    def forward(self, x):
        return self.feed_forward(x)

    def compute_APL(self, X):
        """
        Compute average decision path length given input data. It computes the how many decision nodes one has to
        traverse on average for one data instance.

        Parameters
        -------

        X: Input features

        Returns
        -------

        average decision path lengths, taking the average from several runs with different random seeds

        """

        def node_count(tree):
            return tree.tree_.node_count

        def weighted_node_count(tree, X_train):
            """Weighted node count by example"""
            leaf_indices = tree.apply(X_train)
            leaf_counts = np.bincount(leaf_indices)
            leaf_i = np.arange(tree.tree_.node_count)
            node_count = np.dot(leaf_i, leaf_counts) / float(X_train.shape[0])
            return node_count

        def sequence_to_samples(tensor):
            sequence_array = [tensor[idx, :, :] for idx in range(tensor.shape[0])]
            return np.vstack(sequence_array)

        self.freeze_model()
        self.eval()
        y_tree = self(X)
        y_trxee = torch.where(y_tree > 0.5, 1, 0).detach().cpu().numpy()
        self.unfreeze_model()
        self.train()

        X_tree = X.cpu().detach().numpy()
        tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf)
        tree.fit(X_tree, y_tree)

        return weighted_node_count(tree, X_tree)

    def compute_APL_prediction(self):
        """
        Computes the average-path-length (APL) prediction with the surrogate model using the
        current target model parameters W as input.

        Returns
        -------

        APL prediction as the regulariser Omega(W)
        """
        return self.surrogate_network(self.parameters_to_vector())

    def freeze_model(self):
        """
        Disable model updates by gradient-descent by freezing the model parameters.
        """

        for param in self.feed_forward.parameters():
            param.requires_grad = False

    def freeze_surrogate(self):
        """
        Disable model updates by gradient-descent by freezing the model parameters.
        """

        for param in self.surrogate_network.parameters():
            param.requires_grad = False

    def unfreeze_surrogate(self):
        """
        Enable model updates by gradient-descent by unfreezing the model parameters.
        """
        for param in self.surrogate_network.parameters():
            param.requires_grad = True

    def unfreeze_model(self):
        """
        Enable model updates by gradient-descent by unfreezing the model parameters.
        """
        for param in self.feed_forward.parameters():
            param.requires_grad = True

    def freeze_bias(self):
        """
        Disable model updates by gradient-descent by freezing the biases.
        """
        for name, param in self.feed_forward.named_parameters():
            if 'bias' in name:
                param.requires_grad = False

    def reset_outer_weights(self):
        """
        Reset all weights of the feed forward network for random restarts.
        Required for initial surrogate data.
        """
        self.feed_forward.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    def reset_surrogate_weights(self):
        """
        Reset all weights of the feed forward network for random restarts.
        Required for initial surrogate data.
        """
        self.surrogate_network.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    def parameters_to_vector(self) -> torch.Tensor:
        """
        Convert model parameters to vector.
        """

        return parameters_to_vector(self.feed_forward.parameters())

    def vector_to_parameters(self, parameter_vector):
        """
        Overwrite the model parameters with given parameter vector.
        """
        vector_to_parameters(parameter_vector, self.feed_forward.parameters())

