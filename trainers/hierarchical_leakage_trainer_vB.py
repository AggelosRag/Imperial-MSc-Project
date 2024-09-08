import torch
from sklearn.metrics import accuracy_score

from trainers import IndependentCBMTrainer, JointCBMTrainer
import importlib

from utils.tree_utils import extract_features_from_splits


class HierarchicalLeakageTrainervB:

    def __init__(self, arch, config, device, data_loader,
                 valid_data_loader, reg=None):

        self.arch = arch
        self.config = config
        self.device = device
        self.init_train_data_loader = data_loader
        self.init_valid_data_loader = valid_data_loader
        self.reg = reg

    def train(self):

        # define first expert as independent cbm with selectivenet
        self.first_expert = IndependentCBMTrainer(
            self.arch, self.config, self.device,
            self.init_train_data_loader, self.init_valid_data_loader,
            reg=None, expert=1
        )
        logger = self.config.get_logger('trainer')
        logger.info('\n')
        logger.info('Expert 1 - Training Hard CBM ...')
        print('\nExpert 1 - Training Hard CBM ...')
        self.first_expert.train()
        best_model_path = self.first_expert.cy_epoch_trainer.checkpoint_dir + '/model_best.pth'
        checkpoint = torch.load(best_model_path)
        model_state_dict = checkpoint['state_dict']
        selector_state_dict = checkpoint['selector']
        # self.arch.model.load_state_dict(model_state_dict)
        # self.arch.selector.load_state_dict(selector_state_dict)

        # get the selected train and
        (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train, tree_train,
         tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
            = self.first_expert.cy_epoch_trainer._save_selected_results(
            loader=self.init_train_data_loader, expert=1, mode="train")

        (tensor_X, tensor_C, tensor_y_acc, fi, tree,
        tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
            = self.first_expert.cy_epoch_trainer._save_selected_results(
            loader=self.init_valid_data_loader, expert=1, mode="valid")

        y_pred = tree_train.predict(tensor_C)
        fid = accuracy_score(tensor_y_acc, y_pred)
        print(f'\nAccuracy: {fid}')

        y_pred = tree.predict(tensor_C)
        fid = accuracy_score(tensor_y_acc, y_pred)
        print(f'Accuracy: {fid}')

        # get the feature indices used in the tree of hard cbm
        used_features_ind = extract_features_from_splits(tree_train.tree_)

        # get the config for the first joint expert
        config_joint_cbm = self.config.rest_configs[0]
        self.config._main_config = config_joint_cbm
        logger = self.config.get_logger('trainer')

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_train, tensor_C_train, tensor_y_acc_train)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_train_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True)

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_C, tensor_y_acc)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_valid_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

        logger.info('\n')
        logger.info('Expert 1 - Training Joint CBM ...')
        print('\nExpert 1 - Training Joint CBM ...')

        # build model architecture, then print to console
        arch_module = importlib.import_module("architectures")
        arch = self.config.init_obj('arch', arch_module, self.config, hard_concepts=used_features_ind)
        #arch.scale_concept_weights(fi)
        logger.info('\n')
        logger.info(arch.model)
        reg = config_joint_cbm['regularisation']["type"]

        self.first_joint_expert = JointCBMTrainer(arch, self.config,
                                                  self.device,
                                                  new_train_data_loader,
                                                  new_valid_data_loader,
                                                  reg=reg, iteration=1)
        self.first_joint_expert.epoch_trainer.load_gt_train_tree(tree_train)
        self.first_joint_expert.epoch_trainer.load_gt_val_tree(tree)
        self.first_joint_expert.train()

        # get the config for the second expert
        config_joint_cbm = self.config.rest_configs[1]
        self.config._main_config = config_joint_cbm
        logger = self.config.get_logger('trainer')

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_train_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True)

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_valid_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

        logger.info('\n')
        logger.info('Expert 2 - Training Hard CBM ...')
        print('\nExpert 2 - Training Hard CBM ...')

        # build model architecture, then print to console
        arch_module = importlib.import_module("architectures")
        arch = self.config.init_obj('arch', arch_module, self.config)
        logger.info('\n')
        logger.info(arch.model)
        reg = config_joint_cbm['regularisation']["type"]

        # define second expert as independent cbm with selectivenet
        self.second_expert = IndependentCBMTrainer(
            arch, self.config, self.device, new_train_data_loader,
            new_valid_data_loader, reg=reg, expert=2
        )
        self.second_expert.train()
        best_model_path = self.second_expert.cy_epoch_trainer.checkpoint_dir + '/model_best.pth'
        checkpoint = torch.load(best_model_path)
        model_state_dict = checkpoint['state_dict']
        selector_state_dict = checkpoint['selector']
        # arch.model.load_state_dict(model_state_dict)
        # arch.selector.load_state_dict(selector_state_dict)

        # get the selected train and
        (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train, tree_train,
         tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
            = self.second_expert.cy_epoch_trainer._save_selected_results(
            loader=new_train_data_loader, expert=2, mode="train")

        (tensor_X, tensor_C, tensor_y_acc, fi, tree,
        tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
            = self.second_expert.cy_epoch_trainer._save_selected_results(
            loader=new_valid_data_loader, expert=2, mode="valid")

        y_pred = tree_train.predict(tensor_C)
        fid = accuracy_score(tensor_y_acc, y_pred)
        print(f'\nAccuracy: {fid}')

        y_pred = tree.predict(tensor_C)
        fid = accuracy_score(tensor_y_acc, y_pred)
        print(f'Accuracy: {fid}')

        # get the feature indices used in the tree of hard cbm
        used_features_ind = extract_features_from_splits(tree_train.tree_)

        # get the config for the second joint expert
        config_joint_cbm = self.config.rest_configs[2]
        self.config._main_config = config_joint_cbm
        logger = self.config.get_logger('trainer')

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_train, tensor_C_train, tensor_y_acc_train)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_train_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True)

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_C, tensor_y_acc)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_valid_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

        logger.info('\n')
        logger.info('Expert 2 - Training Joint CBM ...')
        print('\nExpert 2 - Training Joint CBM ...')

        # build model architecture, then print to console
        arch_module = importlib.import_module("architectures")
        arch = self.config.init_obj('arch', arch_module, self.config, hard_concepts=used_features_ind)
        logger.info('\n')
        logger.info(arch.model)
        reg = config_joint_cbm['regularisation']["type"]

        self.second_joint_expert = JointCBMTrainer(arch, self.config,
                                                   self.device,
                                                   new_train_data_loader,
                                                   new_valid_data_loader,
                                                   reg=reg, iteration=2)
        self.second_joint_expert.epoch_trainer.load_gt_train_tree(tree_train)
        self.second_joint_expert.epoch_trainer.load_gt_val_tree(tree)
        self.second_joint_expert.train()

        # get the config for the third expert
        config_joint_cbm = self.config.rest_configs[3]
        self.config._main_config = config_joint_cbm
        logger = self.config.get_logger('trainer')

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_train,
                                                     tensor_C_rej_train,
                                                     tensor_y_rej_train)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_train_data_loader = torch.utils.data.DataLoader(new_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=True)

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_val,
                                                     tensor_C_rej_val,
                                                     tensor_y_rej_val)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_valid_data_loader = torch.utils.data.DataLoader(new_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=False)

        logger.info('\n')
        logger.info('Expert 3 - Training Hard CBM ...')
        print('\nExpert 3 - Training Hard CBM ...')

        # build model architecture, then print to console
        arch_module = importlib.import_module("architectures")
        arch = self.config.init_obj('arch', arch_module, self.config)
        logger.info('\n')
        logger.info(arch.model)
        reg = config_joint_cbm['regularisation']["type"]

        # define secondexpert as independent cbm with selectivenet
        self.third_expert = IndependentCBMTrainer(
            arch, self.config, self.device, new_train_data_loader,
            new_valid_data_loader, reg=reg, expert=3
        )
        self.third_expert.train()
        best_model_path = self.third_expert.cy_epoch_trainer.checkpoint_dir + '/model_best.pth'
        checkpoint = torch.load(best_model_path)
        model_state_dict = checkpoint['state_dict']
        selector_state_dict = checkpoint['selector']
        # arch.model.load_state_dict(model_state_dict)
        # arch.selector.load_state_dict(selector_state_dict)

        # get the selected train and
        (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train, tree_train,
         tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,) \
            = self.third_expert.cy_epoch_trainer._save_selected_results(
            loader=new_train_data_loader, expert=3, mode="train")

        (tensor_X, tensor_C, tensor_y_acc, fi, tree,
         tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
            = self.third_expert.cy_epoch_trainer._save_selected_results(
            loader=new_valid_data_loader, expert=3, mode="valid")

        y_pred = tree_train.predict(tensor_C)
        fid = accuracy_score(tensor_y_acc, y_pred)
        print(f'\nAccuracy: {fid}')

        y_pred = tree.predict(tensor_C)
        fid = accuracy_score(tensor_y_acc, y_pred)
        print(f'Accuracy: {fid}')

        # get the feature indices used in the tree of hard cbm
        used_features_ind = extract_features_from_splits(tree_train.tree_)

        # get the config for the third joint expert
        config_joint_cbm = self.config.rest_configs[4]
        self.config._main_config = config_joint_cbm
        logger = self.config.get_logger('trainer')

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_train,
                                                     tensor_C_train,
                                                     tensor_y_acc_train)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_train_data_loader = torch.utils.data.DataLoader(new_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=True)

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_C,
                                                     tensor_y_acc)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_valid_data_loader = torch.utils.data.DataLoader(new_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=False)

        logger.info('\n')
        logger.info('Expert 3 - Training Joint CBM ...')
        print('\nExpert 3 - Training Joint CBM ...')

        # build model architecture, then print to console
        arch_module = importlib.import_module("architectures")
        arch = self.config.init_obj('arch', arch_module, self.config, hard_concepts=used_features_ind)
        logger.info('\n')
        logger.info(arch.model)
        reg = config_joint_cbm['regularisation']["type"]

        self.third_joint_expert = JointCBMTrainer(arch, self.config,
                                                  self.device,
                                                  new_train_data_loader,
                                                  new_valid_data_loader,
                                                  reg=reg, iteration=3)
        self.third_joint_expert.epoch_trainer.load_gt_train_tree(tree_train)
        self.third_joint_expert.epoch_trainer.load_gt_val_tree(tree)
        self.third_joint_expert.train()
