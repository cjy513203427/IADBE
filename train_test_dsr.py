import logging
from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Dsr
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

datasets = ['screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor',
            'metal_nut', 'bottle', 'hazelnut', 'leather']

for dataset in datasets:
    logger.info(f"================== Processing dataset: {dataset} ==================")
    task = TaskType.SEGMENTATION
    datamodule = MVTec(
        category=dataset,
        image_size=256,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        task=task,
    )

    '''
        latent_anomaly_strength: 0.2
        upsampling_train_ratio: 0.7
    '''
    model = Dsr()

    engine = Engine(
        max_epochs=700,
        pixel_metrics=["AUROC", "PRO"], image_metrics=["AUROC", "PRO"]
    )

    logger.info(f"================== Start training for dataset: {dataset} ==================")
    engine.fit(datamodule=datamodule, model=model)

    logger.info(f"================== Start testing for dataset: {dataset} ==================")
    engine.test(datamodule=datamodule, model=model)
