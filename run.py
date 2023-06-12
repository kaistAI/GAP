import logging
import argparse
from argparse import ArgumentParser
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from Model import Model
from trackers import MetricManager
import math
from pathlib import Path
import os

if __name__ == '__main__':
    # Parsing Arguments
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str, required=True)
    parser.add_argument('--index', default=0, type=int, required=False)
    parser.add_argument('--num_train_epochs', default=15,
                        type=int, required=False)
    parser.add_argument('--check_val_every_n_epoch',
                        default=1, type=int, required=False)
    arg_ = parser.parse_args()

    # Getting configurations
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = argparse.Namespace(**config)

    config.data_index = int(arg_.index)
    config.wandb_run_name = config.wandb_run_name + str(config.data_index)

    # Init configs that are not given
    if 'seed' not in config:
        seed = 42
    if 'cache_dir' not in config:
        config.cache_dir = os.path.join(
            Path.home(), '.cache/huggingface/datasets')
    if 'train_sets' not in config:
        config.train_sets = ""
    if 'valid_sets' not in config:
        config.valid_sets = []
    if 'valid_subset_path' not in config:
        config.valid_subset_path = None
    if 'valid_type_path' not in config:
        config.valid_type_path = None
    if 'learning_rate' not in config:
        config.learning_rate = 5e-5
    if 'loss_fn' not in config:
        config.loss_fn = 'negative'
    if 'gradient_accumulation_steps' not in config:
        config.gradient_accumulation_steps = 1
    if 'num_train_epochs' not in config:
        config.num_train_epochs = arg_.num_train_epochs
    if 'num_workers' not in config:
        config.num_workers = 0
    if 'wandb_log' not in config:
        config.wandb_log = False
    if 'fp16' not in config:
        config.fp16 = False
    if 'check_validation_only' not in config:
        config.check_validation_only = False
    if 'check_val_every_n_epoch' not in config:
        config.check_val_every_n_epoch = arg_.check_val_every_n_epoch
    if 'tokenizer' not in config:
        config.tokenizer_name_or_path = config.model_name_or_path
    if 'target_length' not in config:
        config.target_length = None
    if 'min_train_epochs' not in config:
        config.min_train_epochs = 0
    if 'limit_val_samples' not in config:
        config.limit_val_samples = 0
    if 'save_checkpoint' not in config:
        config.save_checkpoint = False
    if 'decoding_strategy' not in config:
        config.decoding_strategy = 'greedy'

    pl.seed_everything(seed, workers=True)

    # Set console logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s (%(filename)s:%(lineno)d) : %(message)s'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    callbacks = []

    train_set_name = config.train_set.split('/')[-1].split('.')[0]

    metric_averager = MetricManager()
    callbacks.append(metric_averager)

    # Set wandb logger
    if config.wandb_log:
        wandb_logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name)
    else:
        wandb_logger = None

    if config.limit_val_samples:
        limit_val_batches = math.ceil(
            config.limit_val_samples / (config.eval_batch_size * config.ngpu))
    else:
        limit_val_batches = None

    # Setting for pytorch lightning trainer
    train_params = dict(
        accumulate_grad_batches=config.gradient_accumulation_steps,
        accelerator='gpu',
        devices=config.ngpu,
        max_epochs=int(config.num_train_epochs),
        precision=16 if config.fp16 else 32,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_checkpointing=config.save_checkpoint,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        limit_val_batches=limit_val_batches,
        callbacks=callbacks
    )
    trainer = pl.Trainer(**train_params)
    model = Model(config)

    if config.check_validation_only:
        trainer.validate(model)
    else:
        trainer.fit(model)
