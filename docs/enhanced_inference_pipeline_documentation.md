# Enhanced Inference Pipeline Documentation

## Overview

The `enhanced_inference_pipeline.py` is a comprehensive Sentinel-2 satellite imagery processing and analysis tool that extends the basic inference capabilities with advanced metrics computation, bottleneck activation capture, and data export functionality.

## Key Features

- **Comprehensive Metrics**: PSNR, SSIM, SAM (Spectral Angular Mapper), RMSE across all spectral bands
- **Bottleneck Capture**: Extracts and saves encoder bottleneck activations for analysis
- **Input Data Preservation**: Saves original input data alongside processed results
- **Multi-format Export**: Supports both PyTorch (.pt) and NumPy (.npy) formats
- **Advanced Visualization**: RGB composites, band comparisons, and comprehensive plots
- **Performance Benchmarking**: Tracks and visualizes function execution times

## Architecture Overview

```
Input Data (Zarr) → Preprocessing → Model Inference → Metrics Computation
                                        ↓
Bottleneck Capture ← Forward Hooks ← Model Execution
                                        ↓
Data Export ← Results Processing ← Output Generation
```

## Main Components

### 1. Data Ingestion and Preprocessing

#### `preprocess(zarr_path, config, device)`
- Loads Sentinel-2 data from Zarr format
- Chunks data into 320x320 tiles for processing
- Normalizes pixel values and creates validity masks
- Resizes chunks using bilinear interpolation

**Input**: Zarr file path, configuration, device
**Output**: Processed chunks grid, masks grid, metadata

### 2. Model Loading and Inference

#### `load_model(model_cfg, weights_path, device)`
- Initializes U-Net model with EfficientNet-B2 encoder
- Loads pre-trained weights
- Configures model for inference mode

#### `predict_with_metrics_and_bottleneck(model, x_tensor, masks_tensor, bands, device)`
- **Core Function**: Executes model inference with comprehensive analysis
- Captures bottleneck activations using PyTorch forward hooks
- Computes multi-spectral metrics across all bands
- Preserves original input data for export
- **Returns**: predictions, metrics, bottleneck activations, input data

### 3. Metrics Computation

#### MultiSpectralMetrics Integration
- **PSNR**: Peak Signal-to-Noise Ratio for image quality assessment
- **SSIM**: Structural Similarity Index for perceptual similarity
- **SAM**: Spectral Angular Mapper for spectral similarity
- **RMSE**: Root Mean Square Error for reconstruction accuracy

Each metric is computed per spectral band with validity mask handling.

### 4. Data Export and Persistence

#### `save_bottleneck_and_input_data(bottleneck_tensor, input_tensor, zarr_path, output_dir, save_format)`
- Saves both bottleneck activations and original input data
- Supports multiple formats: 'numpy', 'torch', 'both'
- Generates comprehensive statistics and logging
- **File Naming Convention**:
  - `{product_name}_bottleneck.pt/npy`
  - `{product_name}_input.pt/npy`

#### `save_metrics_to_csv(metrics, zarr_path, output_dir)`
- Exports computed metrics to CSV format
- Includes per-band metrics with product metadata
- Provides summary statistics across all bands

### 5. Visualization and Analysis

#### `create_comprehensive_comparison_plot(input_data, output_data, bands, output_dir, sample_idx, chunk_idx)`
- Creates detailed comparison visualizations
- Shows input vs output with difference maps
- Includes per-band RMSE and MAE statistics
- Focuses on specific chunk elements (e.g., element 9)

#### `plot_rgb_composite(reconstruction, bands, output_path)`
- Generates true-color RGB composite images
- Uses B04 (Red), B03 (Green), B02 (Blue) bands
- Provides visual quality assessment

## Workflow Execution

### 1. Initialization
```python
env = initialize_env()  # Load credentials
model_cfg = load_config("config.yaml")  # Model configuration
model = load_model(model_cfg, weights_path, device)  # Load model
```

### 2. Data Processing
```python
chunks_grid, masks_grid, meta = preprocess(zarr_path, config, device)
pred, metrics, bottleneck_tensor, input_data = predict_with_metrics_and_bottleneck(...)
```

### 3. Results Export
```python
csv_path = save_metrics_to_csv(metrics, zarr_path, output_dir)
bottleneck_paths, input_paths = save_bottleneck_and_input_data(...)
```

### 4. Visualization
```python
create_comprehensive_comparison_plot(...)  # Chunk analysis
plot_rgb_composite(...)  # RGB composite
plot_benchmark_results(...)  # Performance analysis
```

## Output Structure

```
inference_results/
├── {product_name}_metrics.csv           # Comprehensive metrics
├── {product_name}_bottleneck.pt/npy     # Bottleneck activations
├── {product_name}_input.pt/npy          # Original input data
├── rgb_composite.png                    # True-color visualization
├── benchmark_results.svg                # Performance metrics
└── chunk_comparisons/                   # Detailed chunk analysis
    └── band_comparison_{band}.png
```

## Performance Benchmarking

The pipeline includes comprehensive performance tracking:
- Function execution times with `@benchmark` decorator
- Memory usage statistics for tensors
- Comparison with traditional Sen2Cor processing times
- Visual benchmark results in log-scale plots

## Key Technical Specifications

### Model Architecture
- **Encoder**: EfficientNet-B2 (pre-trained)
- **Decoder**: U-Net architecture with skip connections
- **Input**: Multi-spectral Sentinel-2 bands
- **Output**: Reconstructed spectral bands

### Data Formats
- **Input**: Zarr format with chunked structure
- **Processing**: PyTorch tensors (CUDA/CPU)
- **Export**: PyTorch (.pt) and NumPy (.npy) formats
- **Visualization**: PNG/SVG for plots

### Bottleneck Capture
- **Location**: `model[0].encoder.blocks[5]` (adjustable)
- **Method**: PyTorch forward hooks
- **Storage**: Detached CPU tensors for memory efficiency

## Usage Example

```python
# Run the complete pipeline
python enhanced_inference_pipeline.py

# Results automatically saved to inference_results/
# - Comprehensive metrics CSV
# - Bottleneck and input data in dual formats
# - Visual comparisons and RGB composites
# - Performance benchmarks
```

## Dependencies

### Core Libraries
- **PyTorch**: Model inference and tensor operations
- **NumPy**: Numerical computations
- **xarray/Zarr**: Satellite data handling
- **Matplotlib**: Visualization
- **Pandas**: Data export and analysis

### Custom Modules
- **model_zoo.models**: Model definitions
- **training.metrics**: MultiSpectralMetrics implementation
- **utils.torch**: Model loading utilities
- **utils.utils**: Configuration management

## Integration Points

The pipeline is designed for integration with:
- **Decoder Analysis**: Bottleneck data feeds into decoder-only pipelines
- **Batch Processing**: Handles multiple Sentinel-2 products
- **Research Workflows**: Comprehensive data export for further analysis
- **Quality Assessment**: Automated metrics computation and reporting

## Performance Characteristics

- **Processing Speed**: Significantly faster than traditional Sen2Cor (35+ minutes)
- **Memory Efficient**: Chunked processing with automatic cleanup
- **Scalable**: Supports variable input sizes and batch processing
- **Quality Metrics**: Comprehensive evaluation across multiple dimensions

This enhanced pipeline serves as a complete solution for Sentinel-2 satellite imagery processing, combining efficient inference with detailed analysis capabilities and robust data export functionality.