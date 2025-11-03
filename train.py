import os
from os.path import join as opj
from datetime import datetime
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# 假设所有模块都在正确的位置
try:
    from utils.misc import reproduc
    from utils.logger import MyLogger # 如果您想使用原版的 Logger
    from dataset.lorenz_datamodule import Lorenz96DataModule
    from lightning_module import CUTSPlusLightning
except ImportError:
    print("Error: Make sure to run this script from the 'cuts_plus_lightning' directory.")
    exit(1)


def main():
    # 1. 加载配置
    # 注意：将 'opt' 目录重命名为 'configs'
    config_path = "configs/lorenz_example.yaml"
    opt = OmegaConf.load(config_path)
    
    # 设置设备/加速器
    has_gpu = torch.cuda.is_available()
    accelerator = "gpu" if has_gpu else "cpu"
    print(f"Using accelerator: {accelerator}")

    # 2. 设置可复现性
    reproduc(**opt.reproduc)

    # 3. 设置 Logger
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
    opt.task_name += timestamp
    proj_path = opj(opt.dir_name, opt.task_name)
    
    # 使用 Lightning 的 TensorBoardLogger
    logger = TensorBoardLogger(save_dir=opt.dir_name, name=opt.task_name, default_hp_metric=False)
    
    # (可选) 如果仍想使用 MyLogger，需要修改 lightning_module.py 中的日志记录方式
    # log = MyLogger(log_dir=proj_path, **opt.log)
    # log.log_opt(opt)
    # logger = log # 但 MyLogger 不是一个 pl.LoggerBase
    
    print(f"Logs will be saved to: {proj_path}")

    # 4. 初始化 DataModule
    # 我们将训练配置和数据配置都传递给它
    datamodule = Lorenz96DataModule(
        data_config=opt.data,
        train_config=opt.sota.cuts_plus,
        reproduc_config=opt.reproduc
    )
    
    print("Setting up datamodule...")
    datamodule.setup('fit')
    
    # 5. 初始化 LightningModule
    model = CUTSPlusLightning(
        train_config=opt.sota.cuts_plus,
        reproduc_config=opt.reproduc
    )
    
    # 6. 初始化 Trainer
    trainer = pl.Trainer(
        max_epochs=opt.sota.cuts_plus.total_epoch,
        logger=logger,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
        check_val_every_n_epoch=opt.sota.cuts_plus.show_graph_every, # 复用这个参数
        enable_checkpointing=False,
        deterministic=opt.reproduc.deterministic,
    )

    # 7. 启动训练
    # datamodule.setup('fit') 会在这里被自动调用
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()