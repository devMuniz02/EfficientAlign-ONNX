# Efficientalign Onnx

> A complete end-to-end pipeline to Fine-tune (QLoRA), Align (DPO/RLHF), and Deploy (ONNX) Large Language Models on consumer hardware. Designed for low resource environments.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/devMuniz02/EfficientAlign-ONNX)](https://github.com/devMuniz02/EfficientAlign-ONNX/issues)
[![GitHub stars](https://img.shields.io/github/stars/devMuniz02/EfficientAlign-ONNX)](https://github.com/devMuniz02/EfficientAlign-ONNX/stargazers)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## ✨ Features

- **QLoRA Fine-tuning**: Efficient fine-tuning of large language models using Quantized Low-Rank Adaptation
- **DPO Alignment**: Direct Preference Optimization for aligning model responses with human preferences
- **ONNX Export**: Convert trained models to ONNX format for optimized inference
- **Low Resource Optimization**: Designed to run on consumer hardware with ~12GB VRAM
- **End-to-End Pipeline**: Complete workflow from training to deployment
- **Hugging Face Integration**: Seamless pushing and loading from Hugging Face Hub

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Git
- Hugging Face account (for pushing models to Hub)
- ~12GB VRAM GPU recommended (for training)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/devMuniz02/EfficientAlign-ONNX.git

# Navigate to the project directory
cd EfficientAlign-ONNX

# Install dependencies
pip install -r requirements.txt
```



```
EfficientAlign-ONNX/
├── assets/                 # Static assets (images, icons, etc.)
├── data/                   # Data files and datasets
├── docs/                   # Documentation files
├── final_model/            # Trained LoRA adapter model
├── merged_model/           # Merged base model + adapter
├── onnx_model/             # Exported ONNX model
├── notebooks/              # Jupyter notebooks for testing
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── train_dpo.py        # DPO training script
│   ├── export_and_push_onnx.py  # ONNX export script
│   ├── push_merged_model.py     # Push merged model to Hub
│   ├── push_to_hub.py      # Hub utilities
│   └── test_hf_and_onnx.py # Testing scripts
├── tests/                  # Unit tests
├── LICENSE                 # License file
├── README.md               # Project documentation
├── README_ONNX.md          # ONNX model usage guide
└── requirements.txt        # Python dependencies
```

## 📁 Project Structure

- **`assets/`**: Static files like images and media
- **`data/`**: Datasets and data-related resources
- **`docs/`**: Additional documentation and guides
- **`final_model/`**: Trained LoRA adapter with config and tokenizer
- **`merged_model/`**: Full merged model ready for inference
- **`onnx_model/`**: ONNX-optimized model for efficient deployment
- **`notebooks/`**: Jupyter notebooks for testing and demonstrations
- **`scripts/`**: Utility scripts for automation
- **`src/`**: Main source code including training and export scripts
- **`tests/`**: Unit tests and test files

## 📖 Usage

### Training Pipeline

1. **Fine-tune with DPO**:
```bash
python src/train_dpo.py
```

2. **Merge LoRA adapter with base model**:
```bash
python src/push_merged_model.py
```

3. **Export to ONNX**:
```bash
python src/export_and_push_onnx.py
```

### Testing Models

- **Test ONNX model**: Run the notebook `notebooks/testsonnx.ipynb`
- **Compare HF vs ONNX**: Use `src/test_hf_and_onnx.py`

### Basic Inference with ONNX Model

See [README_ONNX.md](README_ONNX.md) for detailed ONNX inference examples.

## ⚙️ Configuration

### Environment Variables

Create a `.env` file for Hugging Face authentication:

```bash
HF_TOKEN=your_huggingface_token_here
```

### Training Configuration

The training script uses the following default settings:
- Model: `google/gemma-3-1b-it`
- Dataset: `HuggingFaceH4/ultrafeedback_binarized`
- LoRA rank: 16
- Batch size: Configurable in `DPOConfig`
- Max sequence length: 1024

Modify the scripts in `src/` to adjust parameters for your use case.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For development, also install optional packages
pip install ipython jupyterlab

# Run tests
python -m pytest tests/

# Format code
pip install black
black src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Links:**
- **GitHub:** [https://github.com/devMuniz02/](https://github.com/devMuniz02/)
- **LinkedIn:** [https://www.linkedin.com/in/devmuniz](https://www.linkedin.com/in/devmuniz)
- **Hugging Face:** [https://huggingface.co/manu02](https://huggingface.co/manu02)
- **Portfolio:** [https://devmuniz02.github.io/](https://devmuniz02.github.io/)

Project Link: [https://github.com/devMuniz02/EfficientAlign-ONNX](https://github.com/devMuniz02/EfficientAlign-ONNX)

---

⭐ If you find this project helpful, please give it a star!
