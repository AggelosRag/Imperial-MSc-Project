from sklearn.tree import DecisionTreeClassifier
from torch import nn
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class ParabolaArchitecture:
    def __init__(self, config):
        self.model = TreeNet(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config["model"]['lr'],
                                          weight_decay=config["model"]['weight_decay'])
        self.optimizer_sr = torch.optim.Adam(self.model.surrogate_network.parameters(),
                                             lr=1e-3)
        self.optimizer_mn = torch.optim.Adam(self.model.feed_forward.parameters(),
                                             lr=1e-3)
        self.criterion = torch.nn.MSELoss()
        self.criterion_sr = torch.nn.MSELoss()

class SurrogateNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            #nn.Dropout(0.05),
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
        self.train_size = config["dataset"]["train_size"]

        if self.output_dim == 2:
            network_out = 1
            surrogate_input_dim = self.train_size
        else:
            network_out = self.output_dim
            surrogate_input_dim = self.train_size * self.output_dim

        self.feed_forward = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            #nn.Dropout(0.1),
            nn.Linear(100, 10),
            nn.Tanh(),
            nn.Linear(10, network_out),
        )

        self.surrogate_network = SurrogateNetwork(surrogate_input_dim)
        self.random_seeds = np.random.randint(1, 100, 10)

    def forward(self, x):
        x = self.feed_forward(x)
        if self.output_dim == 2:
            x = torch.sigmoid(x)
        else:
            x = torch.softmax(x, 1)
        return x

    def compute_APL(self, X, X_test, y_test):
        """
        Compute average decision path length given input data. It computes the how many decision nodes one has to
        traverse on average for one data instance.

        Parameters
        -------

        X_test
        y_test
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
        if self.args.output_dim == 2:
            y_tree = torch.sigmoid(y_tree)
            y_tree = np.where(y_tree > 0.5, 1, 0)

            # xx, yy = np.linspace(0, 1, 100), np.linspace(
            #     0, 1, 100)
            # xx, yy = np.meshgrid(xx, yy)
            # Z, data = pred_contours(xx, yy, self, device)
        else:
            y_tree = torch.softmax(y_tree, 1)
            y_tree = np.argmax(y_tree, 1)
        self.unfreeze_model()
        self.train()

        X_tree = X.cpu().detach().numpy()
        # tree.fit(data, Z)

        # ccp_alpha = post_pruning(X_tree, y_tree, self.min_samples_leaf)
        tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf,
                                      ccp_alpha=0)
        # tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        # tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf)
        tree.fit(X_tree, y_tree)

        # path_length = 0
        # for x in X_tree:
        #     path_length += tree.decision_path([x]).toarray().sum()
        #
        # return path_length / len(X_tree)
        # return weighted_node_count(tree, X_tree)
        return node_count(tree)

        # path_lengths = []
        #
        # for random_state in self.random_seeds:
        #     # alphas = post_pruning(X_test, y_test)
        #     alphas = post_pruning(X_tree, y_tree)
        #     tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf,
        #                                   ccp_alpha=alphas,
        #                                   random_state=random_state)
        #     # tree.fit(data, Z)
        #     tree.fit(X_tree, y_tree)
        #     # average_path_length = np.mean(np.sum(tree.tree_.decision_path(np.float32(data)), axis=1))
        #     average_path_length = np.mean(np.sum(tree.tree_.decision_path(X_tree), axis=1))
        #     # average_path_length = weighted_node_count(tree, X_tree)
        #     # average_path_length = weighted_node_count(tree, np.float32(data))
        #     path_lengths.append(average_path_length)
        #
        #     del tree
        #
        # return np.mean(path_lengths)

        # path_lengths = []
        #
        # for random_state in self.random_seeds:
        #     tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf, random_state=random_state)
        #     tree.fit(X_tree, y_tree)
        #     average_path_length = np.mean(np.sum(tree.tree_.decision_path(X_tree), axis=1))
        #     path_lengths.append(average_path_length)
        #
        #     del tree
        #
        # return np.mean(path_lengths)

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

