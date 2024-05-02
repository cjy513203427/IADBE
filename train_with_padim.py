from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Padim

datamodule = MVTec(num_workers=0)
datamodule.prepare_data()  # Downloads the datasets if it's not in the specified `root` directory
datamodule.setup()  # Create train/val/test/prediction sets.
datasets=['screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor', 'metal_nut', 'bottle', 'hazelnut', 'leather']


model = Padim()

for dataset in datasets:
    datamodule = MVTec(
        category=dataset,
        image_size=256,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        task=TaskType.SEGMENTATION,
    )

    engine = Engine(task=TaskType.SEGMENTATION)
    engine.fit(model=model, datamodule=datamodule)

    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
    )