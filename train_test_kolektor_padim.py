import logging
from anomalib import TaskType
from anomalib.data import Kolektor
from anomalib.engine import Engine
from anomalib.models import Padim

# configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dataset = "Kolektor"

logger.info(f"================== Processing dataset: {dataset} ==================")
model = Padim()
datamodule = Kolektor(train_batch_size=32,
                      eval_batch_size=32)
# metrics is under "anomalib/metrics/"
engine = Engine(pixel_metrics=["AUROC", "PRO"], image_metrics=["AUROC", "PRO"], task=TaskType.SEGMENTATION)

logger.info(f"================== Start training for dataset: {dataset} ==================")
engine.fit(model=model, datamodule=datamodule)

logger.info(f"================== Start testing for dataset: {dataset} ==================")
test_results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
)
