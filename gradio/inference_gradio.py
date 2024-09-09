import gradio as gr
from anomalib.engine import Engine
from pathlib import Path

# Import all possible model classes
from anomalib.models import (
    Cfa,
    Cflow,
    Csflow,
    Dfkde,
    Dfm,
    Draem,
    Dsr,
    EfficientAd,
    Fastflow,
    Fre,
    Ganomaly,
    Padim,
    Patchcore,
    ReverseDistillation,
    Rkde,
    Stfpm,
    Uflow,
    AiVad,
    WinClip,
)

# Mapping model filename prefixes to corresponding classes
model_mapping = {
    "Cfa": Cfa,
    "Cflow": Cflow,
    "Csflow": Csflow,
    "Dfkde": Dfkde,
    "Dfm": Dfm,
    "Draem": Draem,
    "Dsr": Dsr,
    "EfficientAd": EfficientAd,
    "Fastflow": Fastflow,
    "Fre": Fre,
    "Ganomaly": Ganomaly,
    "Padim": Padim,
    "Patchcore": Patchcore,
    "ReverseDistillation": ReverseDistillation,
    "Rkde": Rkde,
    "Stfpm": Stfpm,
    "Uflow": Uflow,
    "AiVad": AiVad,
    "WinClip": WinClip,
}

# Define the inference function
def predict(image_path, model_path):
    # Initialize the engine
    engine = Engine(
        pixel_metrics="AUROC",
        accelerator="auto",
        devices=1,
        logger=False,
    )

    # Get the model filename prefix to determine the model type
    model_filename = Path(model_path).stem  # Get the filename without extension
    model_type = model_filename.split("_")[0]  # Use the first part of the filename as the model type

    # Select the corresponding model class based on the filename
    model_class = model_mapping.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}. Please ensure the model file name is correct.")

    # Initialize the model
    model = model_class()

    # Get the image filename
    image_filename = Path(image_path).name

    # Dynamically set the result save path, replacing "Padim" with the extracted model type
    result_dir = Path(f"results/{model_type}/latest/images")
    result_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Perform inference
    engine.predict(
        data_path=image_path,
        model=model,
        ckpt_path=model_path,
    )

    result_path = result_dir / image_filename
    return str(result_path)


# Function to clear input fields
def clear_inputs():
    return None, None


# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Inference/Prediction")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="filepath")
            model_input = gr.File(label="Upload Model File")
            with gr.Row():
                predict_button = gr.Button("Run Inference")
                clear_button = gr.Button("Clear Inputs")

        with gr.Column(scale=3):  # Increase the right column scale
            output_image = gr.Image(label="Output Image", elem_id="output_image", width="100%", height=600)  # Set height

    # Click the inference button to run the prediction function and output the result
    predict_button.click(
        predict,
        inputs=[image_input, model_input],
        outputs=output_image
    )

    # Click the clear button to clear input fields
    clear_button.click(
        clear_inputs,
        outputs=[image_input, model_input]
    )

# Launch the Gradio app
demo.launch()
