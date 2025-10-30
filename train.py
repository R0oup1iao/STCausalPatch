import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import sys
from data.dataset import TimeSeriesDataModule
from lightning_module import CUTSPlusLightning

def train(config_path: str, seed: int = 42):
    """
    主训练函数
    """
    # 0. 固定随机种子
    pl.seed_everything(seed, workers=True)
    
    # 1. 加载配置
    cfg = OmegaConf.load(config_path)
    
    # 2. 初始化 DataModule
    datamodule = TimeSeriesDataModule(
        data_path=cfg.data.data_path,
        mask_path=cfg.data.mask_path,
        config=cfg.data
    )
    
    # 3. 初始化 LightningModule
    model = CUTSPlusLightning(cfg)
    
    # 4. 初始化 Logger
    logger = TensorBoardLogger("logs/", name="CUTS_Plus_Refactor")
    
    # 5. 初始化 Trainer
    trainer = pl.Trainer(
        logger=logger,
        **cfg.trainer # 自动从 config 加载所有 trainer 参数
    )
    
    # 6. 开始训练
    print("--- 开始训练 ---")
    trainer.fit(model, datamodule)
    print("--- 训练完成 ---")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.yaml"
    
    # 你可以从命令行传入种子
    train(config_path, seed=42)