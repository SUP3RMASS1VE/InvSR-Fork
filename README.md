# InvSR: Diffusion-Based Super-Resolution

## Description
InvSR is a diffusion-based super-resolution model that enhances low-resolution images into high-quality, high-resolution images. This implementation leverages **Gradio** for a user-friendly interface and supports both **single image enhancement** and **batch processing**.

## Features
- **Single Image Processing**: Upload an image and enhance it with a diffusion-based approach.
- **Batch Processing**: Process multiple images from a directory at once.
- **Customizable Parameters**: Adjust processing steps, chopping size, and random seed.
- **Interactive Gradio UI**: Intuitive interface for easy image enhancement.

## Installation
Ensure you have Python installed (recommended version: Python 3.8 or higher). Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### Running the Gradio Interface
To start the application, run:

```bash
python app.py
```

This will launch a **Gradio** web interface where you can upload images and apply super-resolution processing.

## Parameters Explained
- **Number of Steps**: Defines the number of timesteps used in diffusion-based processing. More steps typically improve quality but increase computation time.
- **Chopping Size**: Determines the patch size for image processing, useful for handling large images with limited memory.
- **Random Seed**: Allows for reproducibility of results by setting a fixed seed value.

## File Structure
```
InvSR/
├── configs/                # Configuration files
├── weights/                # Model checkpoints
├── utils/                  # Utility scripts
├── app.py                  # Main Gradio application
├── sampler_invsr.py        # Super-resolution sampling
├── requirements.txt        # Required dependencies
└── README.md               # This documentation
```

## License
This project is licensed under **S-Lab License 1.0**. Redistribution and use for **non-commercial purposes** are permitted under specific conditions.

For commercial use, please contact the contributors.

## Credits
Developed by **S-Lab**. Special thanks to **OAOA/InvSR** for their work on diffusion-based super-resolution models.

## Acknowledgments
- **Gradio** for UI implementation.
- **Hugging Face** for model hosting and checkpoint management.
- **tqdm** for progress visualization.

For more details, visit the [original repository](https://github.com/zsyOAOA/InvSR).

