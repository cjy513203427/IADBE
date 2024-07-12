from anomalib import TaskType
from anomalib.data import BTech
from anomalib.engine import Engine
from anomalib.models import Padim

datamodule = BTech(
    root="../datasets/Btech",
    category="01",
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
    ckpt_path="../results/Padim/BTech/01/latest/weights/lightning/model.ckpt",
)