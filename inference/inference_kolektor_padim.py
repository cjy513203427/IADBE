from anomalib.data import Kolektor
from anomalib.engine import Engine
from anomalib.models import Padim
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from anomalib import TaskType

datamodule = Kolektor(
    root="../datasets/Kolektor",
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
    ckpt_path="../results/Padim/Kolektor/latest/weights/lightning/model.ckpt",
)