from torch import nn
import torch

from model.loss import SelectiveNetLoss, CELoss
from networks.model_cub import get_model


class CUBCBMArchitecture:
    def __init__(self, config, hard_concepts=None):

        if hard_concepts is None:
            self.hard_concepts = []
        else:
            self.hard_concepts = hard_concepts

        concept_size = config["dataset"]["num_concepts"] - len(self.hard_concepts)
        self.concept_predictor, _, _ = get_model("/Users/gouse/PycharmProjects/AR-Imperial-Thesis/networks")
        self.label_predictor = LabelPredictor(concept_size=config["dataset"]["num_concepts"],
                                              num_classes=config["dataset"]["num_classes"])
        self.model = MainNetwork(self.concept_predictor, self.label_predictor)

        # Define loss functions and optimizers
        self.criterion_concept = nn.BCEWithLogitsLoss(reduction='none')  # BCE Loss for binary concepts
        #self.concept_weights = torch.ones(config["dataset"]["num_concepts"])
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


class LabelPredictor(nn.Module):
    def __init__(self, concept_size, num_classes):
        super(LabelPredictor, self).__init__()
        # self.fc1 = nn.Linear(concept_size, 100)
        # self.fc2 = nn.Linear(100, 100)
        # self.fc3 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100, num_classes)
        self.fc1 = nn.Linear(concept_size, num_classes)

    def forward(self, c):
        # c = torch.relu(self.fc1(c))
        # c = torch.relu(self.fc2(c))
        # c = torch.relu(self.fc3(c))
        # c = self.fc4(c)
        c = self.fc1(c)
        return c