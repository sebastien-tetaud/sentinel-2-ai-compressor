# Sentinel-2 AI Compressor

A deep learning pipeline for compressing and reconstructing Sentinel-2 satellite data for 11 bands using U-Net like architecture using Timm with comprehensive quality metrics and analysis tools.

## Features

- **AI compression** of Sentinel-2 multispectral imagery
- **Comprehensive quality metrics** (PSNR, SSIM, SAM, RMSE)
- **Bottleneck analysis** with activation capture
- **Enhanced inference pipeline** with data export capabilities
- **Visualization tools** for quality assessment

## Quick Start

### 1. Installation

```bash
git clone git@github.com:sebastien-tetaud/sentinel-2-ai-compressor.git
cd sentinel-2-ai-compressor
conda create -n ai_compressor python==3.13.2
conda activate ai_compressor
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your credentials:
```bash
ACCESS_KEY_ID=your_username
SECRET_ACCESS_KEY=your_password
```

### 3. Usage

**Download Dataset:**
```bash
cd src/generate_dataset
python download_v4.py
```

**Train Model:**
```bash
python main.py
```

**Run Enhanced Inference:**
```bash
cd src
python enhanced_inference_pipeline.py
```

## Key Scripts

- `main.py` - Training pipeline with configuration from `src/cfg/config.yaml`
- `src/inference.py` - Advanced inference with metrics and data export
- `src/notebook/` - Analysis notebooks for data visualization and decoder extraction

## Output

The enhanced inference pipeline generates:
- Quality metrics (CSV format)
- Bottleneck activations (.pt/.npy)
- Original input data (.pt/.npy)
- RGB composite visualizations
- Performance benchmarks

## Model

- **Architecture**: U-Net with EfficientNet-B2 encoder
- **Input**: Multi-spectral Sentinel-2 bands
- **Output**: Reconstructed imagery with quality assessment