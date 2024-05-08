import logging
from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Draem
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
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
        task=task,
    )

    '''
        enable_sspcab: bool = False,
        sspcab_lambda: float = 0.1,
        anomaly_source_path: str | None = None,
        beta: float | tuple[float, float] = (0.1, 1.0),
    '''
    model = Draem()

    callbacks = [
        ModelCheckpoint(
            mode="max",
            monitor="pixel_AUROC",
        ),
        EarlyStopping(
            monitor="pixel_AUROC",
            mode="max",
            patience=20,
        ),
    ]

    engine = Engine(
        max_epochs=700,
        callbacks=callbacks,
        pixel_metrics=["AUROC", "PRO"], image_metrics=["AUROC", "PRO"],
        accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
        devices=1,
        logger=False,
    )

    logger.info(f"================== Start training for dataset: {dataset} ==================")
    engine.fit(datamodule=datamodule, model=model)

    logger.info(f"================== Start testing for dataset: {dataset} ==================")
    engine.test(datamodule=datamodule, model=model)
