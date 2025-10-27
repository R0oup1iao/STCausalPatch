import torch
import argparse
import os
from omegaconf import OmegaConf
import numpy as np

# 
from utils.logging import log_string
from utils.data_loader import load_dataset # 
from utils.spatial import HierarchicalKDTreePartitioner
from models.stcp_net import STCPNet
from trainer import STCPTrainer

# === 全新重构的代码 ===
# 

def run(config, log_f):
    # 1. 
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    log_string(log_f, f"Using device: {device}")

    # 2. 
    log_string(log_f, "Loading data...")
    dataset_pack = load_dataset(config, log_f)

    # 3. 
    config.model.n_nodes = dataset_pack['n_nodes']
    log_string(log_f, f"Number of nodes set to: {config.model.n_nodes}")

    # 4. 
    partitioner = None 
    if dataset_pack['locations'] is not None:
        log_string(log_f, "Initializing KDTree Partitioner (spatial data found).")
        partitioner = HierarchicalKDTreePartitioner(
            locations=dataset_pack['locations'],
            config=config.spatial
        )
        # 
        if not config.model.use_spatial_emb:
             log_string(log_f, "Warning: Spatial data found, but use_spatial_emb=False in config.")
    else:
        log_string(log_f, "Skipping Partitioner (no spatial data found).")
        # 
        if config.model.use_spatial_emb:
            config.model.use_spatial_emb = False
            log_string(log_f, "Warning: No spatial data found. Forcing use_spatial_emb = False.")


    # 5. 
    model = STCPNet(config).to(device)

    # 6. 
    trainer = STCPTrainer(
        config, model, partitioner, 
        dataset_pack, device, log_f
    )

    # 7. 
    if config.mode == 'train':
        log_string(log_f, "Starting training...")
        trainer.train()
        log_string(log_f, "Training finished. Starting testing...")
        trainer.test()
    elif config.mode == 'test':
        log_string(log_f, "Starting testing...")
        trainer.test()

    log_string(log_f, "Run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stcp_real_world.yaml", help='configuration file')
    args = parser.parse_args()

    # 
    config = OmegaConf.load(args.config)

    # 
    config.mode = config.get('mode', 'train') 

    # 
    if config.data.dataset_type == 'synthetic':
        log_name = f"STCP_Synthetic_{config.synthetic.data_type}_{config.synthetic.n_nodes}N.log"
    else:
        # 
        data_name = os.path.basename(config.data.traffic_file).split('.')[0]
        log_name = f"STCP_Real_{data_name}.log"

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)

    log_file_path = os.path.join(config.log_dir, log_name)
    log_f = open(log_file_path, 'w') # 

    log_string(log_f, '------------ Options -------------')
    log_string(log_f, OmegaConf.to_yaml(config))
    log_string(log_f, '-------------- End ----------------')

    try:
        run(config, log_f)
    except Exception as e:
        log_string(log_f, f"An error occurred: {e}")
        import traceback
        log_string(log_f, traceback.format_exc())
    finally:
        log_f.close()