# Import the necessary modules
from anomalib.data import Avenue
from anomalib.engine import Engine
from anomalib.models import AiVad

# Load the avenue dataset, model and engine.
datamodule = Avenue()
model = AiVad()
engine = Engine()

# Train the model
engine.train(model, datamodule)