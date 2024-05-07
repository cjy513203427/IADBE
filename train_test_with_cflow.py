from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Cflow
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

datasets = ['screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor',
            'metal_nut', 'bottle', 'hazelnut', 'leather']

for dataset in datasets:
    task = TaskType.SEGMENTATION
    datamodule = MVTec(
        category=dataset,
        image_size=256,
        train_batch_size=128,
        eval_batch_size=128,
        num_workers=0,
        task=task,
    )

    '''
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3", "layer4"),
        pre_trained: bool = True,
        fiber_batch_size: int = 64,
        decoder: str = "freia-cflow",
        condition_vector: int = 128,
        coupling_blocks: int = 8,
        clamp_alpha: float = 1.9,
        permute_soft: bool = False,
        lr: float = 0.0001,
    '''
    model = Cflow()

    callbacks = [
        ModelCheckpoint(
            mode="max",
            monitor="pixel_AUROC",
        ),
        EarlyStopping(
            monitor="pixel_AUROC",
            mode="max",
            patience=2,
        ),
    ]

    engine = Engine(
        max_epochs=50,
        callbacks=callbacks,
        pixel_metrics=["AUROC", "PRO"],
        accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
        devices=1,
        logger=False,
    )

    engine.fit(datamodule=datamodule, model=model)

    engine.test(datamodule=datamodule, model=model)
