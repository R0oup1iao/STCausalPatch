import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import sys
from dataset.dataset import TimeSeriesDataModule
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
        config=cfg.data
    )
    
    # 3. (移动) 初始化 Logger
    logger = TensorBoardLogger("logs/", name="CUTS_Plus_Refactor")

    # 4. (移动) 初始化 Trainer
    trainer = pl.Trainer(
        logger=logger,
        **cfg.trainer # 自动从 config 加载所有 trainer 参数
    )

    # 5. (新) 手动调用 datamodule.setup()
    # 这将确保在 'simulate' 模式下，n_nodes 和 data_dim 被正确设置
    # (在 trainer.fit() 内部也会调用，但我们提前调用以确保配置正确)
    print("--- 准备数据 ---")
    trainer.fit_loop.setup_data() 
    # (注意: 上面这行是较新版 PL 的 API，如果报错)
    # (请使用: datamodule.setup('fit'))

    # 6. (新) 检查配置是否被数据覆盖
    if datamodule.cfg.n_nodes != cfg.data.n_nodes:
        print(f"配置 n_nodes 被数据覆盖: {cfg.data.n_nodes} -> {datamodule.cfg.n_nodes}")
        cfg.data.n_nodes = datamodule.cfg.n_nodes
        cfg.model.n_nodes = datamodule.cfg.n_nodes # (如果模型配置也需要)
        cfg.causal.n_groups_start = datamodule.cfg.n_nodes # (假设无分组)

    if datamodule.cfg.data_dim != cfg.data.data_dim:
        print(f"配置 data_dim 被数据覆盖: {cfg.data.data_dim} -> {datamodule.cfg.data_dim}")
        cfg.data.data_dim = datamodule.cfg.data_dim

    # 7. (移动) 初始化 LightningModule
    # (在 datamodule.setup() 之后初始化，以使用正确的 n_nodes)
    print("--- 初始化模型 ---")
    model = CUTSPlusLightning(cfg)

    # 8. (移动) 开始训练
    print("--- 开始训练 ---")
    # (我们已经调用了 setup_data, fit 会直接开始)
    trainer.fit(model, datamodule=datamodule)
    print("--- 训练完成 ---")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "./config/lorenz_96.yaml"
    
    # 你可以从命令行传入种子
    train(config_path, seed=42)