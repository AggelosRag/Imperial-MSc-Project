import argparse
import collections
import torch
import numpy as np
from trainers.leakage_inspection_general_case import perform_leakage_visualization
from utils.parse_config import ConfigParser, _update_config
from utils import prepare_device
import importlib


# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.random.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    dataloaders_module = importlib.import_module("data_loaders")
    data_loader, valid_data_loader, test_data_loader = config.init_obj(
        'data_loader', dataloaders_module, config=config
    )

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    # build model architecture, then print to console
    arch_module = importlib.import_module("architectures")
    arch = config.init_obj('arch', arch_module, config=config, device=device,
                           data_loader=data_loader)
    logger.info("\n")
    logger.info(arch.model)

    if 'explainer' in config.config.keys():
        perform_leakage_visualization(data_loader, arch, config)
        return 0
    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    # metrics = [getattr(module_metric, met) for met in config['metrics']]
    #metrics = config['metrics']
    reg = config['regularisation']["type"]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainers = importlib.import_module("trainers")
    trainer = config.init_obj('trainer', trainers,
                              arch=arch,
                              config=config,
                              device=device,
                              data_loader=data_loader,
                              valid_data_loader=valid_data_loader,
                              reg=reg)

    trainer.train()
    print("\nTraining completed")
    print("Starting testing ...")
    logger.info("\n")
    logger.info("Training completed")

    if config["trainer"]['type'] == 'IndependentCBMTrainer':
        hard_cbm = config["trainer"]['hard_cbm']
    else:
        hard_cbm = False

    logger.info("Starting testing ...")
    trainer.test(test_data_loader, hard_cbm=hard_cbm)
    print("\nTesting completed")
    logger.info("\n")
    logger.info("Testing completed")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Imperial Diploma Project')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
