from sklearn.tree import DecisionTreeClassifier
from torch import nn
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from data_loaders import find_class_imbalance_mnist
from networks.custom_dt_gini_with_entropy_metrics import \
    CustomDecisionTree
from model.loss import SelectiveNetLoss, CELoss

class MNISTBlackBoxArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        self.model = ConceptPredictor(config["dataset"]["num_classes"])

        # Define loss functions and optimizers
        self.criterion_label = CELoss()

        if "weight_decay" not in config["model"]:
            xc_params_to_update = [
                {'params': self.model.concept_predictor.parameters(), 'weight_decay': config["model"]['xc_weight_decay']},
            ]
            self.xc_optimizer = torch.optim.Adam(xc_params_to_update, lr=config["model"]['xc_lr'])
            cy_params_to_update = [
                {'params': self.model.label_predictor.parameters(), 'weight_decay': config["model"]['cy_weight_decay']},
            ]
            self.cy_optimizer = torch.optim.Adam(cy_params_to_update, lr=config["model"]['cy_lr'])
        else:
            # only apply regularisation (if any) to the label predictor,
            # for a fair comparison with Tree Regularisation
            params_to_update = [
                {'params': self.model.parameters(),
                 'weight_decay': config["model"]['weight_decay']},
            ]
            self.optimizer = torch.optim.Adam(params_to_update,
                                              lr=config["model"]['lr'])
        self.lr_scheduler = None

class MNISTCBMwithDTaslabelPredictorArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        if hard_concepts is None:
            self.hard_concepts = []
        else:
            self.hard_concepts = hard_concepts

        C_train = data_loader.dataset[:][1]
        if "use_attribute_imbalance" in config["dataset"]:
            if config["dataset"]["use_attribute_imbalance"]:
                self.imbalance = torch.FloatTensor(find_class_imbalance_mnist(C_train)).to(device)
            else:
                self.imbalance = None
        else:
            self.imbalance = None

        self.concept_predictor = ConceptPredictor(config["dataset"]["num_concepts"])
        # self.label_predictor = DecisionTreeClassifier(min_samples_leaf=config["regularisation"]["min_samples_leaf"])
        self.label_predictor = CustomDecisionTree(min_samples_leaf=config["regularisation"]["min_samples_leaf"],
                                                  n_classes=config["dataset"]["num_classes"])
        self.model = MainNetwork(self.concept_predictor, self.label_predictor)

        if "pretrained_concept_predictor" in config["model"]:
            state_dict = torch.load(config["model"]["pretrained_concept_predictor"])["state_dict"]
            # Create a new state dictionary for the concept predictor layers
            concept_predictor_state_dict = {}

            # Iterate through the original state dictionary and isolate concept predictor layers
            for key, value in state_dict.items():
                if key.startswith('concept_predictor'):
                    # Remove the prefix "concept_predictor."
                    new_key = key.replace('concept_predictor.', '')
                    concept_predictor_state_dict[new_key] = value

            self.model.concept_predictor.load_state_dict(concept_predictor_state_dict)
            print("Loaded pretrained concept predictor from ", config["model"]["pretrained_concept_predictor"])

        # Define loss functions and optimizers
        self.criterion_concept = torch.nn.BCEWithLogitsLoss(weight=self.imbalance)
        self.criterion_per_concept = nn.BCEWithLogitsLoss(reduction='none')

        xc_params_to_update = [
            {'params': self.model.concept_predictor.parameters(), 'weight_decay': config["model"]['xc_weight_decay']},
        ]
        self.xc_optimizer = torch.optim.Adam(xc_params_to_update, lr=config["model"]['xc_lr'])
        self.cy_optimizer = None
        self.lr_scheduler = None

class MNISTCBMTreeArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        if hard_concepts is None:
            self.hard_concepts = []
        else:
            self.hard_concepts = hard_concepts

        C_train = data_loader.dataset[:][1]
        if "use_attribute_imbalance" in config["dataset"]:
            if config["dataset"]["use_attribute_imbalance"]:
                self.imbalance = torch.FloatTensor(find_class_imbalance_mnist(C_train)).to(device)
            else:
                self.imbalance = None
        else:
            self.imbalance = None


        concept_size = config["dataset"]["num_concepts"] - len(self.hard_concepts)
        self.concept_predictor = ConceptPredictor(concept_size)
        self.sr_model = SurrogateNetwork(
            input_dim=config["dataset"]["train_size"] * config["dataset"]["num_classes"]
        )
        self.label_predictor = LabelPredictor(concept_size=config["dataset"]["num_concepts"],
                                              num_classes=config["dataset"]["num_classes"])
        self.mn_model = MainNetwork(self.concept_predictor, self.label_predictor)
        self.model = TreeNet(self.mn_model, self.sr_model)

        # Define loss functions and optimizers
        self.criterion_concept = torch.nn.BCEWithLogitsLoss(weight=self.imbalance)
        self.criterion_per_concept = nn.BCEWithLogitsLoss(reduction='none')  # BCE Loss for binary concepts        self.criterion_label = CELoss()
        self.criterion_sr = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config["model"]['lr'],
                                          weight_decay=config["model"]['weight_decay'])
        self.optimizer_sr = torch.optim.Adam(self.sr_model.parameters(), lr=0.001)
        self.optimizer_mn = torch.optim.Adam(self.mn_model.parameters(), lr=0.001)

class MNISTCBMArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        if hard_concepts is None:
            self.hard_concepts = []
        else:
            self.hard_concepts = hard_concepts

        C_train = data_loader.dataset[:][1]
        if "use_attribute_imbalance" in config["dataset"]:
            if config["dataset"]["use_attribute_imbalance"]:
                self.imbalance = torch.FloatTensor(find_class_imbalance_mnist(C_train)).to(device)
            else:
                self.imbalance = None
        else:
            self.imbalance = None

        concept_size = config["dataset"]["num_concepts"] - len(self.hard_concepts)
        self.concept_predictor = ConceptPredictor(concept_size)
        self.label_predictor = LabelPredictor(concept_size=config["dataset"]["num_concepts"],
                                              num_classes=config["dataset"]["num_classes"])
        self.model = MainNetwork(self.concept_predictor, self.label_predictor)

        if "pretrained_concept_predictor" in config["model"]:
            state_dict = torch.load(config["model"]["pretrained_concept_predictor"])["state_dict"]
            # Create a new state dictionary for the concept predictor layers
            concept_predictor_state_dict = {}

            # Iterate through the original state dictionary and isolate concept predictor layers
            for key, value in state_dict.items():
                if key.startswith('concept_predictor'):
                    # Remove the prefix "concept_predictor."
                    new_key = key.replace('concept_predictor.', '')
                    concept_predictor_state_dict[new_key] = value

            self.model.concept_predictor.load_state_dict(concept_predictor_state_dict)
            print("Loaded pretrained concept predictor from ", config["model"]["pretrained_concept_predictor"])

        # Define loss functions and optimizers
        self.criterion_concept = torch.nn.BCEWithLogitsLoss(weight=self.imbalance)
        self.criterion_per_concept = nn.BCEWithLogitsLoss(reduction='none')
        self.criterion_label = CELoss()

        if "weight_decay" not in config["model"]:
            xc_params_to_update = [
                {'params': self.model.concept_predictor.parameters(), 'weight_decay': config["model"]['xc_weight_decay']},
            ]
            self.xc_optimizer = torch.optim.Adam(xc_params_to_update, lr=config["model"]['xc_lr'])
            cy_params_to_update = [
                {'params': self.model.label_predictor.parameters(), 'weight_decay': config["model"]['cy_weight_decay']},
            ]
            self.cy_optimizer = torch.optim.Adam(cy_params_to_update, lr=config["model"]['cy_lr'])
        else:
            # only apply regularisation (if any) to the label predictor,
            # for a fair comparison with Tree Regularisation
            params_to_update = [
                {'params': self.model.concept_predictor.parameters(),
                 'weight_decay': 0},
                {'params': self.model.label_predictor.parameters(),
                 'weight_decay': config["model"]['weight_decay']},
            ]
            self.optimizer = torch.optim.Adam(params_to_update,
                                              lr=config["model"]['lr'])

        self.lr_scheduler = None

class MNISTCBMSelectiveNetArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        if hard_concepts is None:
            self.hard_concepts = []
        else:
            self.hard_concepts = hard_concepts

        C_train = data_loader.dataset[:][1]
        if "use_attribute_imbalance" in config["dataset"]:
            if config["dataset"]["use_attribute_imbalance"]:
                self.imbalance = torch.FloatTensor(find_class_imbalance_mnist(C_train)).to(device)
            else:
                self.imbalance = None
        else:
            self.imbalance = None

        concept_size = config["dataset"]["num_concepts"] - len(self.hard_concepts)
        self.concept_predictor = ConceptPredictor(concept_size)
        self.label_predictor = LabelPredictor(concept_size=config["dataset"]["num_concepts"],
                                              num_classes=config["dataset"]["num_classes"])
        self.model = MainNetwork(self.concept_predictor, self.label_predictor)

        # define the selector network
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(config["dataset"]["num_concepts"], config["dataset"]["num_concepts"]),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(config["dataset"]["num_concepts"]),
            torch.nn.Linear(config["dataset"]["num_concepts"], 1),
            torch.nn.Sigmoid(),
        )
        self.aux_model = LabelPredictor(concept_size=config["dataset"]["num_concepts"],
                                        num_classes=config["dataset"]["num_classes"])

        # Define loss functions and optimizers
        self.criterion_concept = torch.nn.BCEWithLogitsLoss(weight=self.imbalance)
        self.criterion_per_concept = nn.BCEWithLogitsLoss(reduction='none')
        CE = nn.CrossEntropyLoss(reduction='none')
        self.criterion_label = SelectiveNetLoss(
            iteration=1, CE=CE,
            selection_threshold=config["selectivenet"]["selection_threshold"],
            coverage=config["selectivenet"]["coverage"],
            lm=config["selectivenet"]["lm"], dataset=config["dataset"],
            alpha=config["selectivenet"]["alpha"]
        )

        if "weight_decay" not in config["model"]:
            xc_params_to_update = [
                {'params': self.model.concept_predictor.parameters(), 'weight_decay': config["model"]['xc_weight_decay']},
            ]
            self.xc_optimizer = torch.optim.Adam(xc_params_to_update, lr=config["model"]['xc_lr'])
            cy_params_to_update = [
                {'params': self.model.label_predictor.parameters(), 'weight_decay': config["model"]['cy_weight_decay']},
                {'params': self.selector.parameters(), 'weight_decay': config["model"]['cy_weight_decay']},
                {'params': self.aux_model.parameters(), 'weight_decay': config["model"]['cy_weight_decay']},
            ]
            self.cy_optimizer = torch.optim.Adam(cy_params_to_update, lr=config["model"]['cy_lr'])
        else:
            # only apply regularisation (if any) to the label predictor,
            # for a fair comparison with Tree Regularisation
            params_to_update = [
                {'params': self.model.concept_predictor.parameters(), 'weight_decay': 0},
                {'params': self.model.label_predictor.parameters(), 'weight_decay': config["model"]['weight_decay']},
                {'params': self.selector.parameters(), 'weight_decay': config["model"]['weight_decay']},
                {'params': self.aux_model.parameters(), 'weight_decay': config["model"]['weight_decay']},
            ]
            self.optimizer = torch.optim.Adam(params_to_update,
                                              lr=config["model"]['lr'])

class MNISTCYArchitecture:
    def __init__(self, config):

        self.model = LabelPredictor(concept_size=config["dataset"]["num_features"],
                                    num_classes=config["dataset"]["num_classes"])

        # Define loss functions and optimizers
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config["model"]['lr'],
                                          weight_decay=config["model"]['weight_decay'])

# Define the models
# class ConceptPredictor(nn.Module):
#     def __init__(self, num_concepts):
#         super(ConceptPredictor, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, num_concepts)
#
#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         #x = self.fc3(x)
#         x = self.fc3(x)  # Sigmoid activation for binary concepts
#         return x

class ConceptPredictor(nn.Module):
    def __init__(self, num_concepts):
        super(ConceptPredictor, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # Convolutional layer
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(256, 120),
            # Adjust the input size based on your CNN output
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_concepts)  # Output 6 regression values
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.fc_model(x)
        #x = torch.sigmoid(self.fc_model(x))  # Sigmoid activation for binary concepts
        return x

class LabelPredictor(nn.Module):
    def __init__(self, concept_size, num_classes):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(concept_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, num_classes)

    def forward(self, c):
        c = torch.relu(self.fc1(c))
        c = torch.relu(self.fc2(c))
        c = torch.relu(self.fc3(c))
        c = self.fc4(c)
        return c

class MainNetwork(nn.Module):
    def __init__(self, concept_predictor, label_predictor):
        super(MainNetwork, self).__init__()

        self.concept_predictor = concept_predictor
        self.label_predictor = label_predictor

    def unfreeze_model(self):
        """
        Enable model updates by gradient-descent by unfreezing the model parameters.
        """
        for param in self.parameters():
            param.requires_grad = True

    def freeze_model(self):
        """
        Disable model updates by gradient-descent by freezing the model parameters.
        """
        for param in self.parameters():
            param.requires_grad = False

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
    def __init__(self, mn_model, sr_model):
        super(TreeNet, self).__init__()

        self.mn_model = mn_model
        self.sr_model = sr_model

    def unfreeze_model(self):
        """
        Enable model updates by gradient-descent by unfreezing the model parameters.
        """
        for param in self.mn_model.parameters():
            param.requires_grad = True

    def freeze_model(self):
        """
        Disable model updates by gradient-descent by freezing the model parameters.
        """
        for param in self.mn_model.parameters():
            param.requires_grad = False

    def freeze_surrogate(self):
        """
        Disable model updates by gradient-descent by freezing the model parameters.
        """

        for param in self.sr_model.parameters():
            param.requires_grad = False

    def unfreeze_surrogate(self):
        """
        Enable model updates by gradient-descent by unfreezing the model parameters.
        """
        for param in self.sr_model.parameters():
            param.requires_grad = True
