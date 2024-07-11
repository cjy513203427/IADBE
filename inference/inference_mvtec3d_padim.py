from anomalib import TaskType
from anomalib.data import MVTec3D
from anomalib.engine import Engine
from anomalib.models import Padim

# select category "carrot" of MVTec3D
datamodule = MVTec3D(
    root="../datasets/MVTec3D",
    category="carrot",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0,
    task=TaskType.SEGMENTATION,
)

engine = Engine(
    pixel_metrics="AUROC",
    accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
    devices=1,
    logger=False,
)

model = Padim()
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path="../results/Padim/MVTec3D/carrot/latest/weights/lightning/model.ckpt",
)