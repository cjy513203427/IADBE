import gradio as gr
from anomalib.engine import Engine
from anomalib.models import Padim
from pathlib import Path

# Define the inference function
def predict(image_path, model_path):
    # Initialize the engine
    engine = Engine(
        pixel_metrics="AUROC",
        accelerator="auto",
        devices=1,
        logger=False,
    )

    # Initialize the model
    model = Padim()

    # Get the image file name
    image_filename = Path(image_path).name

    # Define the result save path (using Path object to ensure cross-platform compatibility)
    result_dir = Path("results/Padim/latest/images")
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
