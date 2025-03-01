import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import gradio as gr
from pathlib import Path
from omegaconf import OmegaConf
from sampler_invsr import InvSamplerSR
import os
from tqdm import tqdm

from utils import util_common
from utils import util_image
from basicsr.utils.download_util import load_file_from_url

def get_configs(num_steps=1, chopping_size=128, seed=12345):
    configs = OmegaConf.load("./configs/sample-sd-turbo.yaml")

    if num_steps == 1:
        configs.timesteps = [200,]
    elif num_steps == 2:
        configs.timesteps = [200, 100]
    elif num_steps == 3:
        configs.timesteps = [200, 100, 50]
    elif num_steps == 4:
        configs.timesteps = [200, 150, 100, 50]
    elif num_steps == 5:
        configs.timesteps = [250, 200, 150, 100, 50]
    else:
        assert num_steps <= 250
        configs.timesteps = np.linspace(
            start=250, stop=0, num=num_steps, endpoint=False, dtype=np.int64()
        ).tolist()
    print(f'Setting timesteps for inference: {configs.timesteps}')

    configs.sd_path = "./weights"
    util_common.mkdir(configs.sd_path, delete=False, parents=True)
    configs.sd_pipe.params.cache_dir = configs.sd_path

    started_ckpt_name = "noise_predictor_sd_turbo_v5.pth"
    started_ckpt_dir = "./weights"
    util_common.mkdir(started_ckpt_dir, delete=False, parents=True)
    started_ckpt_path = Path(started_ckpt_dir) / started_ckpt_name
    if not started_ckpt_path.exists():
        load_file_from_url(
            url="https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth",
            model_dir=started_ckpt_dir,
            progress=True,
            file_name=started_ckpt_name,
        )
    configs.model_start.ckpt_path = str(started_ckpt_path)

    configs.bs = 1
    configs.seed = seed
    configs.basesr.chopping.pch_size = chopping_size
    configs.basesr.chopping.extra_bs = 4

    return configs

def predict_single(in_path, num_steps=1, chopping_size=128, seed=12345, progress=gr.Progress()):
    if in_path is None:
        return None, None, "Please upload an image first!"
    
    progress(0, desc="Initializing model...")
    configs = get_configs(num_steps=num_steps, chopping_size=chopping_size, seed=seed)
    sampler = InvSamplerSR(configs)

    progress(0.2, desc="Processing image...")
    out_dir = Path('invsr_output')
    if not out_dir.exists():
        out_dir.mkdir()
    sampler.inference(in_path, out_path=out_dir, bs=1)

    progress(0.9, desc="Finalizing results...")
    out_path = out_dir / f"{Path(in_path).stem}.png"
    assert out_path.exists(), 'Super-resolution failed!'
    im_sr = util_image.imread(out_path, chn="rgb", dtype="uint8")
    
    im_lr = util_image.imread(in_path, chn="rgb", dtype="uint8")
    
    progress(1.0, desc="Done!")
    return im_sr, str(out_path), "✅ Super-resolution completed successfully!"

def process_batch(input_dir, num_steps=1, chopping_size=128, seed=12345, progress=gr.Progress()):
    if not input_dir or not os.path.exists(input_dir):
        return "⚠️ Please provide a valid input directory path."
    
    input_path = Path(input_dir)
    output_path = input_path / 'invsr_output'
    output_path.mkdir(exist_ok=True)

    progress(0.1, desc="Initializing model...")
    configs = get_configs(num_steps=num_steps, chopping_size=chopping_size, seed=seed)
    sampler = InvSamplerSR(configs)

    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + list(input_path.glob('*.jpeg'))
    total_files = len(image_files)

    if total_files == 0:
        return f"⚠️ No image files found in {input_dir}"

    progress(0.2, desc="Processing images")
    for idx, img_path in enumerate(image_files):
        out_path = output_path / f"{img_path.stem}.png"
        sampler.inference(str(img_path), out_path=output_path, bs=1)
        progress((0.2 + 0.8 * (idx + 1)/total_files), desc=f"Processing image {idx + 1}/{total_files}")

    return f"✅ Successfully processed {total_files} images!\n\nResults saved in: {output_path}"

css = """
:root {
    --bg-color: #1a1e24;
    --card-bg: #222831;
    --accent-color: #5d5fef;
    --accent-hover: #4a4cbe;
    --text-color: #e6e6e6;
    --secondary-text: #a0a0a0;
    --border-color: #2c3440;
}

body {
    background-color: var(--bg-color) !important;
    color: var(--text-color) !important;
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg-color) !important;
    color: var(--text-color) !important;
}

.main-div {
    background-color: var(--bg-color);
    border-radius: 12px;
    padding: 20px;
}

.container {
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}

.header {
    text-align: center;
    margin-bottom: 20px;
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-color);
    margin-bottom: 10px;
}

.header p {
    font-size: 1.1rem;
    color: var(--secondary-text);
    margin-bottom: 20px;
}

.tabs {
    margin-top: 20px;
}

.param-container {
    background-color: var(--card-bg);
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    margin-bottom: 15px;
    border: 1px solid var(--border-color);
}

.param-title {
    font-weight: 600;
    margin-bottom: 10px;
    color: var(--text-color);
}

.result-container {
    background-color: var(--card-bg);
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border-color);
}

.btn-primary {
    background-color: var(--accent-color) !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
}

.btn-primary:hover {
    background-color: var(--accent-hover) !important;
}

.status-msg {
    padding: 10px;
    border-radius: 6px;
    font-weight: 500;
    margin-top: 10px;
    color: var(--text-color);
}

.tooltip {
    font-size: 0.9rem;
    color: var(--secondary-text);
    margin-top: 5px;
}

.license-footer {
    text-align: center;
    margin-top: 30px;
    padding: 15px;
    background-color: var(--card-bg);
    border-radius: 8px;
    border: 1px solid var(--border-color);
    font-size: 0.9rem;
    color: var(--secondary-text);
}

.license-footer a {
    color: var(--accent-color);
    text-decoration: none;
}

.license-footer a:hover {
    text-decoration: underline;
}

/* Override Gradio's default styles for dark theme */
.dark input, .dark textarea, .dark select {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-color) !important;
}

.dark label {
    color: var(--text-color) !important;
}

.dark .gr-box, .dark .gr-form, .dark .gr-panel {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
}

.dark .gr-input, .dark .gr-button {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
}

.dark .gr-form {
    background-color: var(--card-bg) !important;
}

.dark .gr-input-label {
    color: var(--text-color) !important;
}

/* Tab styling */
.tabs-header {
    background-color: var(--card-bg) !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.tabs-header button {
    color: var(--secondary-text) !important;
}

.tabs-header button.selected {
    color: var(--accent-color) !important;
    border-bottom-color: var(--accent-color) !important;
}

/* Slider styling */
input[type=range]::-webkit-slider-thumb {
    background: var(--accent-color) !important;
}

input[type=range]::-moz-range-thumb {
    background: var(--accent-color) !important;
}

input[type=range]::-ms-thumb {
    background: var(--accent-color) !important;
}

/* Radio button styling */
input[type=radio]:checked {
    background-color: var(--accent-color) !important;
    border-color: var(--accent-color) !important;
}

/* File upload area */
.upload-box {
    background-color: var(--card-bg) !important;
    border: 2px dashed var(--border-color) !important;
}

/* Image display areas */
.image-preview {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
}
"""

description = """
<div class="header">
    <h1>InvSR: Diffusion-based Super-resolution</h1>
    <p>Transform your low-quality images into high-resolution masterpieces with state-of-the-art diffusion technology</p>
</div>
"""

license_info = """
<div class="license-footer">
    <p>📋 <strong>License</strong></p>
    <p>This project is licensed under <a rel="license" href="https://github.com/zsyOAOA/InvSR/blob/master/LICENSE">S-Lab License 1.0</a>.</p>
    <p>Redistribution and use for non-commercial purposes should follow this license.</p>
</div>
"""

with gr.Blocks(css=css, theme=gr.themes.Base(primary_hue="indigo")) as demo:
    gr.HTML(description)

    with gr.Tabs(elem_classes="tabs") as tabs:
        with gr.Tab("Single Image", elem_id="single-tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Column(elem_classes="param-container"):
                        gr.Markdown("### Input Settings", elem_classes="param-title")
                        input_image = gr.Image(
                            type="filepath", 
                            label="Upload your image", 
                            elem_id="input-image",
                            height=300
                        )
                        gr.Markdown(
                            "Upload a low-resolution image to enhance", 
                            elem_classes="tooltip"
                        )
                    
                    with gr.Column(elem_classes="param-container"):
                        gr.Markdown("### Processing Parameters", elem_classes="param-title")
                        num_steps = gr.Slider(
                            minimum=1, 
                            maximum=5, 
                            value=2, 
                            step=1, 
                            label="Number of steps",
                            info="More steps = better quality but slower processing"
                        )
                        chopping_size = gr.Radio(
                            choices=[128, 256], 
                            value=128, 
                            label="Chopping size",
                            info="Larger size requires more VRAM but may improve quality"
                        )
                        seed = gr.Number(
                            value=12345, 
                            precision=0, 
                            label="Random seed",
                            info="Set a specific seed for reproducible results"
                        )
                        process_btn = gr.Button("Enhance Image", elem_classes="btn-primary")
                        status_msg = gr.Markdown(elem_classes="status-msg")

                with gr.Column(scale=1):
                    with gr.Column(elem_classes="result-container"):
                        gr.Markdown("### Enhanced Result", elem_classes="param-title")
                        output_image = gr.Image(
                            type="numpy", 
                            label="High-Resolution Output",
                            elem_id="output-image",
                            height=300
                        )
                        output_file = gr.File(label="Download enhanced image")

        with gr.Tab("Batch Processing", elem_id="batch-tab"):
            with gr.Column(elem_classes="param-container"):
                gr.Markdown("### Batch Processing Settings", elem_classes="param-title")
                input_dir = gr.Textbox(
                    label="Input Directory Path",
                    placeholder="Enter the full path to your images folder",
                    info="The folder should contain .jpg, .png, or .jpeg files"
                )
                with gr.Row():
                    batch_num_steps = gr.Slider(
                        minimum=1, 
                        maximum=5, 
                        value=2, 
                        step=1, 
                        label="Number of steps",
                        info="More steps = better quality but slower processing"
                    )
                    batch_chopping_size = gr.Radio(
                        choices=[128, 256], 
                        value=128, 
                        label="Chopping size",
                        info="Larger size requires more VRAM but may improve quality"
                    )
                    batch_seed = gr.Number(
                        value=12345, 
                        precision=0, 
                        label="Random seed",
                        info="Set a specific seed for reproducible results"
                    )
                batch_btn = gr.Button("Process All Images", elem_classes="btn-primary")
                output_text = gr.Markdown(label="Processing Status")
    
    gr.HTML(license_info)

    process_btn.click(
        fn=predict_single,
        inputs=[input_image, num_steps, chopping_size, seed],
        outputs=[output_image, output_file, status_msg]
    )

    batch_btn.click(
        fn=process_batch,
        inputs=[input_dir, batch_num_steps, batch_chopping_size, batch_seed],
        outputs=output_text
    )

demo.queue(max_size=10).launch(share=False)
