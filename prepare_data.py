from anomalib.data import MVTec

datamodule = MVTec(num_workers=0)
datamodule.prepare_data()  # Downloads the datasets if it's not in the specified `root` directory
datamodule.setup()  # Create train/val/test/prediction sets.