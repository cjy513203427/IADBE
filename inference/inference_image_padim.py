from anomalib.engine import Engine
from anomalib.models import Padim

engine = Engine(
    pixel_metrics="AUROC",
    accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
    devices=1,
    logger=False,
)

model = Padim()
# After invoking predict, you will get results under the same folder
# Inference according to an image
predictions = engine.predict(
    model=model,
    ckpt_path="../results/Padim/MVTec/bottle/latest/weights/lightning/model.ckpt",
    data_path="/home/jinyao/PycharmProjects/IADBE/datasets/MVTec/cable/test/combined/001.png"
)
