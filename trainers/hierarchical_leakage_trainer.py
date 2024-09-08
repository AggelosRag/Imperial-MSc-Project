import torch

from trainers import IndependentCBMTrainer, JointCBMTrainer
import importlib

class HierarchicalLeakageTrainer:

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
        self.first_expert.train()

        # get the selected train and
        (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train, tree_train,
         tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
            = self.first_expert.cy_epoch_trainer._save_selected_results(
            loader=self.init_train_data_loader, expert=1, mode="train")

        (tensor_X, tensor_C, tensor_y_acc, fi, tree,
        tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
            = self.first_expert.cy_epoch_trainer._save_selected_results(
            loader=self.init_valid_data_loader, expert=1, mode="valid")

        # get the config for the second expert
        config_joint_cbm = self.config.rest_configs[0]
        self.config._main_config = config_joint_cbm
        logger = self.config.get_logger('trainer')

        # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_train, tensor_C_train, tensor_y_acc_train)
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_train_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True)

        # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_C, tensor_y_acc)
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_valid_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

        # build model architecture, then print to console
        arch_module = importlib.import_module("architectures")
        arch = self.config.init_obj('arch', arch_module, self.config)
        #arch.scale_concept_weights(fi)
        logger.info(arch.model)
        reg = config_joint_cbm['regularisation']["type"]

        self.first_joint_expert = JointCBMTrainer(arch, self.config,
                                                  self.device,
                                                  new_train_data_loader,
                                                  new_valid_data_loader,
                                                  reg=reg, iteration=2)
        self.first_joint_expert.train()

        # get the selected train and
        (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train, tree_train,
         tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
            = self.first_joint_expert.epoch_trainer._save_selected_results(
            loader=self.init_train_data_loader, expert=2, mode="train")

        (tensor_X, tensor_C, tensor_y_acc, fi, tree,
        tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
            = self.first_joint_expert.epoch_trainer._save_selected_results(
            loader=self.init_valid_data_loader, expert=2, mode="valid")

        # get the config for the second expert
        config_joint_cbm = self.config.rest_configs[1]
        self.config._main_config = config_joint_cbm
        logger = self.config.get_logger('trainer')

        # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_train, tensor_C_train, tensor_y_acc_train)
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_train_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True)

        # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_C, tensor_y_acc)
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_valid_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

        # build model architecture, then print to console
        arch_module = importlib.import_module("architectures")
        arch = self.config.init_obj('arch', arch_module, self.config)
        #arch.scale_concept_weights(fi)
        logger.info(arch.model)
        reg = config_joint_cbm['regularisation']["type"]

        self.second_joint_expert = JointCBMTrainer(arch, self.config,
                                                   self.device,
                                                   new_train_data_loader,
                                                   new_valid_data_loader,
                                                   reg=reg, iteration=3)
        self.second_joint_expert.train()

        # get the selected train and
        (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train, tree_train,
         tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
            = self.second_joint_expert.epoch_trainer._save_selected_results(
            loader=new_train_data_loader, expert=3, mode="train")

        (tensor_X, tensor_C, tensor_y_acc, fi, tree,
        tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
            = self.second_joint_expert.epoch_trainer._save_selected_results(
            loader=new_valid_data_loader, expert=3, mode="valid")

        # get the config for the second expert
        config_joint_cbm = self.config.rest_configs[2]
        self.config._main_config = config_joint_cbm
        logger = self.config.get_logger('trainer')

        # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_train, tensor_C_train, tensor_y_acc_train)
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_train_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True)

        # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_C, tensor_y_acc)
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val)
        batch_size = config_joint_cbm['data_loader']['args']['batch_size']
        new_valid_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

        # build model architecture, then print to console
        arch_module = importlib.import_module("architectures")
        arch = self.config.init_obj('arch', arch_module, self.config)
        #arch.scale_concept_weights(fi)
        logger.info(arch.model)
        reg = config_joint_cbm['regularisation']["type"]

        self.third_joint_expert = JointCBMTrainer(arch, self.config,
                                                  self.device,
                                                  new_train_data_loader,
                                                  new_valid_data_loader,
                                                  reg=reg, iteration=4)
        self.third_joint_expert.train()

        # get the selected train and
        (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train, tree_train,
         tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
            = self.third_joint_expert.epoch_trainer._save_selected_results(
            loader=new_train_data_loader, expert=4, mode="train")

        (tensor_X, tensor_C, tensor_y_acc, fi, tree,
        tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
            = self.third_joint_expert.epoch_trainer._save_selected_results(
            loader=new_valid_data_loader, expert=4, mode="valid")

        # get the config for the second expert
        # config_ind_cbm = self.config.rest_configs[1]
        # self.config._main_config = config_ind_cbm
        # logger = self.config.get_logger('trainer')
        #
        # # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_train,
        #                                              tensor_C_rej_train,
        #                                              tensor_y_rej_train
        #                                              )
        # batch_size = config_ind_cbm['data_loader']['args']['batch_size']
        # new_train_data_loader = torch.utils.data.DataLoader(new_dataset,
        #                                                     batch_size=batch_size,
        #                                                     shuffle=True)
        #
        # # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_val,
        #                                              tensor_C_rej_val,
        #                                              tensor_y_rej_val)
        # batch_size = config_ind_cbm['data_loader']['args']['batch_size']
        # new_valid_data_loader = torch.utils.data.DataLoader(new_dataset,
        #                                                     batch_size=batch_size,
        #                                                     shuffle=False)
        #
        # # build model architecture, then print to console
        # arch_module = importlib.import_module("architectures")
        # arch = self.config.init_obj('arch', arch_module, self.config)
        # logger.info(arch.model)
        # reg = config_ind_cbm['regularisation']["type"]
        #
        # self.second_expert = IndependentCBMTrainer(
        #     arch, self.config, self.device, new_train_data_loader,
        #     new_valid_data_loader, reg=reg, expert=2
        # )
        # self.second_expert.train()
        #
        # # get the selected train and
        # (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train,
        #  tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
        #     = self.second_expert.cy_epoch_trainer._save_selected_results(
        #     loader = new_train_data_loader,
        #     iteration = 2,
        #     mode = "train"
        # )
        #
        # (tensor_X, tensor_C, tensor_y_acc, fi,
        # tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
        #     = self.second_expert.cy_epoch_trainer._save_selected_results(
        #     loader = new_valid_data_loader,
        #     iteration = 2,
        #     mode = "valid"
        # )
        #
        # # get the config for the second expert
        # config_ind_cbm = self.config.rest_configs[3]
        # self.config._main_config = config_ind_cbm
        # logger = self.config.get_logger('trainer')
        #
        # # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_train,
        #                                              tensor_C_rej_train,
        #                                              tensor_y_rej_train
        #                                              )
        # batch_size = config_ind_cbm['data_loader']['args']['batch_size']
        # new_train_data_loader = torch.utils.data.DataLoader(new_dataset,
        #                                                     batch_size=batch_size,
        #                                                     shuffle=True)
        #
        # # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_val,
        #                                              tensor_C_rej_val,
        #                                              tensor_y_rej_val)
        # batch_size = config_ind_cbm['data_loader']['args']['batch_size']
        # new_valid_data_loader = torch.utils.data.DataLoader(new_dataset,
        #                                                     batch_size=batch_size,
        #                                                     shuffle=False)
        #
        # # build model architecture, then print to console
        # arch_module = importlib.import_module("architectures")
        # arch = self.config.init_obj('arch', arch_module, self.config)
        # logger.info(arch.model)
        # reg = config_ind_cbm['regularisation']["type"]
        #
        # self.third_expert = IndependentCBMTrainer(
        #     arch, self.config, self.device, new_train_data_loader,
        #     new_valid_data_loader, reg=reg, expert=3
        # )
        # self.third_expert.train()
        #
        # # get the selected train and
        # (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train,
        #  tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
        #     = self.third_expert.cy_epoch_trainer._save_selected_results(
        #     loader = new_train_data_loader,
        #     iteration = 3,
        #     mode = "train"
        # )
        #
        # (tensor_X, tensor_C, tensor_y_acc, fi,
        # tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
        #     = self.third_expert.cy_epoch_trainer._save_selected_results(
        #     loader = new_valid_data_loader,
        #     iteration = 3,
        #     mode = "valid"
        # )
        #
        # # get the config for the second expert
        # config_ind_cbm = self.config.rest_configs[5]
        # self.config._main_config = config_ind_cbm
        # logger = self.config.get_logger('trainer')
        #
        # # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_train,
        #                                              tensor_C_rej_train,
        #                                              tensor_y_rej_train
        #                                              )
        # batch_size = config_ind_cbm['data_loader']['args']['batch_size']
        # new_train_data_loader = torch.utils.data.DataLoader(new_dataset,
        #                                                     batch_size=batch_size,
        #                                                     shuffle=True)
        #
        # # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_val,
        #                                              tensor_C_rej_val,
        #                                              tensor_y_rej_val)
        # batch_size = config_ind_cbm['data_loader']['args']['batch_size']
        # new_valid_data_loader = torch.utils.data.DataLoader(new_dataset,
        #                                                     batch_size=batch_size,
        #                                                     shuffle=False)
        #
        # # build model architecture, then print to console
        # arch_module = importlib.import_module("architectures")
        # arch = self.config.init_obj('arch', arch_module, self.config)
        # logger.info(arch.model)
        # reg = config_ind_cbm['regularisation']["type"]
        #
        # self.fourth_expert = IndependentCBMTrainer(
        #     arch, self.config, self.device, new_train_data_loader,
        #     new_valid_data_loader, reg=reg, expert=4
        # )
        # self.fourth_expert.train()
        #
        # # get the selected train and
        # (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train,
        #  tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
        #     = self.fourth_expert.cy_epoch_trainer._save_selected_results(
        #     loader = new_train_data_loader,
        #     iteration = 4,
        #     mode = "train"
        # )
        #
        # (tensor_X, tensor_C, tensor_y_acc, fi,
        # tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
        #     = self.fourth_expert.cy_epoch_trainer._save_selected_results(
        #     loader = new_valid_data_loader,
        #     iteration = 4,
        #     mode = "valid"
        # )
        #
        # # get the config for the second expert
        # config_ind_cbm = self.config.rest_configs[5]
        # self.config._main_config = config_ind_cbm
        # logger = self.config.get_logger('trainer')
        #
        # # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_train,
        #                                              tensor_C_rej_train,
        #                                              tensor_y_rej_train
        #                                              )
        # batch_size = config_ind_cbm['data_loader']['args']['batch_size']
        # new_train_data_loader = torch.utils.data.DataLoader(new_dataset,
        #                                                     batch_size=batch_size,
        #                                                     shuffle=True)
        #
        # # create a new dataloader with the selected samples
        # new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_val,
        #                                              tensor_C_rej_val,
        #                                              tensor_y_rej_val)
        # batch_size = config_ind_cbm['data_loader']['args']['batch_size']
        # new_valid_data_loader = torch.utils.data.DataLoader(new_dataset,
        #                                                     batch_size=batch_size,
        #                                                     shuffle=False)
        #
        # # build model architecture, then print to console
        # arch_module = importlib.import_module("architectures")
        # arch = self.config.init_obj('arch', arch_module, self.config)
        # logger.info(arch.model)
        # reg = config_ind_cbm['regularisation']["type"]
        #
        # self.fifth_expert = IndependentCBMTrainer(
        #     arch, self.config, self.device, new_train_data_loader,
        #     new_valid_data_loader, reg=reg, expert=5
        # )
        # self.fifth_expert.train()
        #
        # # get the selected train and
        # (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train,
        #  tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
        #     = self.fifth_expert.cy_epoch_trainer._save_selected_results(
        #     loader = new_train_data_loader,
        #     iteration = 5,
        #     mode = "train"
        # )
        #
        # (tensor_X, tensor_C, tensor_y_acc, fi,
        # tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
        #     = self.fifth_expert.cy_epoch_trainer._save_selected_results(
        #     loader = new_valid_data_loader,
        #     iteration = 5,
        #     mode = "valid"
        # )