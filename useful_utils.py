import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import pywt
import torchvision
from torchvision import transforms
from sklearn.decomposition import PCA

from src.architectures.cnn import Simple_CNN_1D, AE_CNN_1D
from src.architectures.transformers import TS_Transformer
from src.architectures.resnet2d import ResNet
from src.trainers import SIM_CLR_Trainer, Deep_Clustering_Trainer, AE_Trainer, PCA_Trainer
from src.datasets import DronesDataset, WiSig_Dataset, LoRaDataset
from src.architectures import side_networks
from matplotlib import pyplot as plt
import wandb
import os
import json
import copy

DATASETS_DICT = {
    
    'WiSig': WiSig_Dataset,
    'LoRa': LoRaDataset,
    'Drones': DronesDataset
    
}

def report(metrics, trainer, exp_config, fold_number, iteration, epoch, train_config, test_config):

    """
    Report to wandb and logs dir
    """
    
    root = exp_config['logs_dir']

    dataset_name = exp_config['dataset']['name']

    approach_name = exp_config['approach']['name']

    id = exp_config['exp_id']

    features_exctractor_name = exp_config['feature_extractor']['name']

    log_dir = os.path.join(
        root, 
        f"{id}/{dataset_name}/{fold_number}/{approach_name}/{features_exctractor_name}/{iteration}/{epoch}"
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    trainer.save_checkpoint(log_dir + '/models.pt')
    
    with open(log_dir + '/config.json', 'w') as f:
        json.dump(exp_config, f, indent=4)

    #with open(log_dir + '/train_config.json', 'w') as f:
    #    json.dump(train_config, f, indent=4)

    #with open(log_dir + '/test_config.json', 'w') as f:
    #    json.dump(test_config, f, indent=4)

    with open(log_dir + '/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    wandb.log(
        metrics,
        step = epoch
    )
    

def evauate_config(
    exp_config: dict
):
    """
    Train config and report prerfomance
    """
    train_configs, test_configs, unknown_devices_folds = get_data_configs(exp_config['dataset']['config'])

    dataset_name = exp_config['dataset']['name']

    clusters_numbers = exp_config['evaluation']['clusters_numbers']

    dataset_class = DATASETS_DICT[dataset_name]

    train_loader_config = exp_config['train_loader']
    
    test_loader_config = exp_config['test_loader']

    num_iterations = exp_config['num_iterations']

    starting_iteration = exp_config['starting_iteration']

    report_interval = exp_config['report_interval']
    
    for fold_number, (train_config, test_config, unknown_devices) in enumerate(
        zip(train_configs, test_configs, unknown_devices_folds)) :

        train_dataset = dataset_class(**train_config)
        
        test_dataset = dataset_class(**test_config)

        targets = []

        for i in range(len(test_dataset)):
            _, device = test_dataset[i]
            targets.append(device in unknown_devices)

        targets = np.array(targets)

        train_loader = DataLoader(train_dataset, **train_loader_config)

        test_loader = DataLoader(test_dataset, **test_loader_config)

        for iteration in range(starting_iteration, num_iterations):
            
            trainer = get_trainer(exp_config)


            wandb.init(
                project = f"{dataset_name}_evaluations",
                config = exp_config | train_config | test_config | {'iteration' : iteration} | {'fold': fold_number},
                name = f"fold_{fold_number}, iter_{iteration}, approach_{exp_config['approach']['name']}, f_extractor_{exp_config['feature_extractor']['name']}"
            )
            
            for epoch in range(trainer.num_epochs):
                loss = trainer.train_epoch(train_loader)

                if epoch % report_interval == 0:
                    
                    metrics = trainer.evaluate(
                        train_loader, test_loader, targets, clusters_numbers = clusters_numbers)
                    
                    metrics['loss'] = loss

                    report(metrics, trainer, exp_config, fold_number, iteration, epoch, train_config, test_config)
                    
            wandb.finish()

                    
                

def get_data_configs(dataset_config):

    """
    Obtain lists of train and test configs using cross validation
    """

    if 'k_fold' in dataset_config.keys():
        
        ratio = dataset_config['k_fold']['ratio']
        
        total_devices = dataset_config['k_fold']['total_devices']

    else:
        
        dataset_config_train = copy.deepcopy(dataset_config)
        dataset_config_train['devices'] = dataset_config_train['train']
        del dataset_config_train['train']

        dataset_config_test = copy.deepcopy(dataset_config)
        dataset_config_test['devices'] = dataset_config_test['test']
        del dataset_config_train['test']

        unknwn_devices = [
            i for i in dataset_config_test['devices'] 
            if not i in dataset_config_train['devices']
        ]

        return [dataset_config_train], [dataset_config_test], [unknwn_devices]

    test_devices = tuple(np.arange(total_devices))
    
    test_config = copy.deepcopy(dataset_config)
    
    test_config.pop('k_fold')
    
    test_config['devices'] = test_devices

    test_configs = [test_config] * ratio

    train_configs = []
    
    moving_part = total_devices // ratio

    unknown_devices_folds = []

    for i in range(ratio):
        
        train_devices = tuple(
            list(np.arange(moving_part * i)) + list(np.arange(moving_part * (i+1), total_devices))
        )

        unknown_devices = np.arange(moving_part * i, moving_part * (i+1))

        unknown_devices_folds.append(unknown_devices)
        
        train_config = copy.deepcopy(dataset_config)
    
        train_config.pop('k_fold')

        train_config['devices'] = train_devices
        
        train_configs.append(train_config)

    return train_configs, test_configs, unknown_devices_folds
    


def get_optimizer(models, optimizer_name, optimizer_config):
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(models.parameters(), **optimizer_config)
    
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(models.parameters(), **optimizer_config)
    
    return optimizer



def get_trainer(
        exp_config : dict
):
    """
    Build trainer object
    """
    approach_name =   exp_config['approach']['name']
    approach_config = exp_config['approach']['config']
    
    feature_extractor_name =   exp_config['feature_extractor']['name']
    feature_extractor_config = exp_config['feature_extractor']['config']

    trainer_config = exp_config['approach']['trainer']


    if feature_extractor_name == 'Simple_CNN_1D':
        
        feature_extractor = Simple_CNN_1D(**feature_extractor_config)
    
    if feature_extractor_name == 'ResNet_1D':
        
        feature_extractor = ResNet_1D(**feature_extractor_config)

    if feature_extractor_name == 'AE_KANS':
        
        feature_extractor = KANS_AE(**feature_extractor_config)

    if feature_extractor_name == 'AE_CNN_1D':
        
        feature_extractor = AE_CNN_1D(**feature_extractor_config)

    
    main_oprimizer_name =   trainer_config['main_optimizer']['name']
    main_optimizer_config = trainer_config['main_optimizer']['config']
    
    
    if approach_name == 'Sim_CLR':
        
        mlp_head_config = trainer_config['mlp_head']

        mlp_instance = side_networks.Mlp(**mlp_head_config)

        models = nn.ModuleDict({
                    'feature_extractor': feature_extractor,
                    'mlp_instance': mlp_instance,
        })

        main_optimizer = get_optimizer(
                    models,
                    main_oprimizer_name,
                    main_optimizer_config)

        optimzers = {'main_optimizer': main_optimizer}

        models['augs'] = side_networks.get_augmentations(
            **trainer_config['augmentations'], type = approach_config['augs_type'])

        if approach_config['large_augs']:
            
            models['large_augs'] = side_networks.get_augmentations(**trainer_config['large_augmentations'], type = 'large_augs')
            

        if approach_config['clusters_loss']:
            
            mlp_cluster = side_networks.Mlp(**mlp_head_config, apply_softmax = True)
            
            models['mlp_cluster'] = mlp_cluster
        

        if approach_config['augs_type'] == 'learnable':
            
            if approach_config['large_augs']:
                
                optimzers['augs_optimizer'] = get_optimizer(
                            nn.ModuleList([models['augs'], models['large_augs']]),
                            trainer_config['augs_optimizer']['name'],
                            trainer_config['augs_optimizer']['config'])
            else:
                
                optimzers['augs_optimizer'] = get_optimizer(
                    models['augs'],
                    trainer_config['augs_optimizer']['name'],
                    trainer_config['augs_optimizer']['config']
                )

        trainer = SIM_CLR_Trainer(
            optimizers = optimzers,
            models = models,
            **approach_config)

    if approach_name == 'Deep Clustering':

        models = nn.ModuleDict({    
            'feature_extractor': feature_extractor,
        })

        main_optimizer = get_optimizer(
                    models,
                    main_oprimizer_name,
                    main_optimizer_config
        )

        optimizers = {'main_optimizer': main_optimizer}

        trainer = Deep_Clustering_Trainer(
            optimizers = optimizers,
            models = models,
            **approach_config)

    if approach_name == 'PCA':
        
        pca = PCA(feature_extractor_config)
        
        trainer = PCA_Trainer(**approach_config)

    if approach_name == 'AE':
        
        models = nn.ModuleDict({    
            'feature_extractor': feature_extractor,
        })

        main_optimizer = get_optimizer(
                    models,
                    main_oprimizer_name,
                    main_optimizer_config)

        optimizers = {'main_optimizer': main_optimizer}

        trainer = AE_Trainer(optimizers = optimizers,
                             models = models,
                             **approach_config)

    return trainer

def parse_configs(exp_configs):
    """
    Go through the config and 
    create all possible combimations of configs with fixed parameters.
    """
    
    cur_node = exp_configs
    stack = [cur_node]

    while len(stack) > 0:
        cur_node = stack.pop()
        for child in cur_node.keys():
            if type(cur_node[child]) == list:
                values = cur_node[child]
                configs = []
                for value in values:
                    cur_node[child] = value
                    configs += parse_configs(copy.deepcopy(exp_configs))
                return configs
                    
            if type(cur_node[child]) == dict:
                stack.append(cur_node[child])
        
    return [copy.deepcopy(exp_configs)]
        
    
    
    