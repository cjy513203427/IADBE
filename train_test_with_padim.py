from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Padim

datasets = ['screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor',
            'metal_nut', 'bottle', 'hazelnut', 'leather']

for dataset in datasets:
    model = Padim()
    datamodule = MVTec(category=dataset, num_workers=0)
    # metrics is under "anomalib/metrics/"
    engine = Engine(pixel_metrics=["AUROC", "PRO"], task=TaskType.SEGMENTATION)
    engine.fit(model=model, datamodule=datamodule)

    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
    )
