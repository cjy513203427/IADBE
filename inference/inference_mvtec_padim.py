from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Padim

# select category "bottle" of MVTec
datamodule = MVTec(
    root="../datasets/MVTec",
    category="bottle",
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

'''
Alternatively you can use the following models
    "Cfa",
    "Cflow",
    "Csflow",
    "Dfkde",
    "Dfm",
    "Draem",
    "Dsr",
    "EfficientAd",
    "Fastflow",
    "Fre",
    "Ganomaly",
    "Padim",
    "Patchcore",
    "ReverseDistillation",
    "Rkde",
    "Stfpm",
    "Uflow",
    "AiVad",
    "WinClip",
'''
model = Padim()
# After invoking predict, you will get results under the same folder
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path="../results/Padim/MVTec/bottle/latest/weights/lightning/model.ckpt",
)
