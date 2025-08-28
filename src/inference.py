import io
import os
import sys
import time
import warnings
from functools import wraps
import random

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
import zarr
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

# Project imports
from model_zoo.models import define_model
from utils.torch import load_model_weights
from utils.utils import load_config
from training.metrics import MultiSpectralMetrics

# Optional imports that might not be available
try:
    import pystac_client
    PYSTAC_AVAILABLE = True
except ImportError:
    PYSTAC_AVAILABLE = False
    logger.warning("pystac_client not available - data query functions will be disabled")

try:
    from auth.auth import S3Connector
    from utils.stac_client import get_product_content
    from utils.utils import extract_s3_path_from_url
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logger.warning("S3 utilities not available - S3 functions will be disabled")

warnings.filterwarnings('ignore')

# Global store for function durations
function_durations = {}

def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Store duration in global dictionary
        function_durations[func.__name__] = function_durations.get(func.__name__, []) + [duration]

        logger.info(f"[BENCHMARK] {func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper

@benchmark
def initialize_env(key_id=None, secret_key=None) -> dict:
    """Load environment variables safely."""
    try:
        load_dotenv()

        # Use provided parameters first, then command line args, then environment variables
        if key_id is None and len(sys.argv) > 1:
            key_id = sys.argv[1]
        if secret_key is None and len(sys.argv) > 2:
            secret_key = sys.argv[2]

        # Fall back to environment variables
        if key_id is None:
            key_id = os.getenv('ACCESS_KEY_ID')
        if secret_key is None:
            secret_key = os.getenv('SECRET_ACCESS_KEY')

        if not key_id or not secret_key:
            logger.warning("Missing credentials - some functionality may be limited")

        logger.success("Environment variables loaded")
        return {
            "access_key_id": str(key_id) if key_id else "",
            "secret_access_key": str(secret_key) if secret_key else ""
        }
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return {"access_key_id": "", "secret_access_key": ""}

def remove_last_segment_rsplit(sentinel_id):
    # Split from the right side, max 1 split
    parts = sentinel_id.rsplit('_', 1)
    return parts[0]

@benchmark
def connect_to_s3(endpoint_url: str, access_key_id: str, secret_access_key: str) -> tuple:
    """Connect to S3 storage."""
    if not S3_AVAILABLE:
        logger.error("S3 utilities not available - cannot connect to S3")
        return None, None

    try:
        connector = S3Connector(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region_name='default'
        )
        logger.success(f"Successfully connected to {endpoint_url}")
        return connector.get_s3_resource(), connector.get_s3_client()
    except Exception as e:
        logger.error(f"Failed to connect to S3 storage: {e}")
        return None, None

@benchmark
def data_query(catalog, bbox: list, start_date: str, end_date: str, max_cloud_cover: int):
    """
    Fetch L2A products from CDSE STAC catalog.

    Args:
        catalog: STAC catalog client
        bbox: Bounding box coordinates [west, south, east, north]
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        max_cloud_cover: Maximum cloud cover percentage

    Returns:
        str: Product URL or None if no suitable products found
    """
    if not PYSTAC_AVAILABLE:
        logger.error("pystac_client not available - cannot perform data query")
        return None

    try:
        # Search for L2A products
        logger.info(f"Searching for L2A products from {start_date} to {end_date} in bbox {bbox}")
        l2a_items = catalog.search(
            collections=['sentinel-2-l2a'],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=1000,
        ).item_collection()

        # Filter L2A items to remove those with high nodata percentage
        l2a_items = [item for item in l2a_items if item.properties.get("statistics", {}).get('nodata', 100) < 5]

        if not l2a_items:
            logger.warning("No suitable L2A items found")
            return None

        selected_item = random.choice(l2a_items)
        product_url, _ = os.path.split(selected_item.assets['safe_manifest'].href)
        logger.info(f"Selected product: {os.path.basename(product_url)}")
        return product_url

    except Exception as e:
        logger.error(f"Error fetching Sentinel data: {e}")
        return None


@benchmark
def download_sentinel_data(env, config, product_url, dir_path):
    """Download Sentinel data and convert to Zarr format."""
    try:
        from eopf.common.constants import OpeningMode
        from eopf.common.file_utils import AnyPath
        from eopf.store.convert import convert
    except ImportError:
        logger.error("eopf modules not available - cannot download Sentinel data")
        return None

    if not env.get('access_key_id') or not env.get('secret_access_key'):
        logger.error("Missing credentials - cannot download data")
        return None

    try:
        ACCESS_KEY_ID = env['access_key_id']
        SECRET_ACCESS_KEY = env['secret_access_key']

        tmp_dir = os.path.join(dir_path, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        zarr_filename = os.path.split(product_url)[1].replace('.SAFE', '.zarr')
        zarr_path = os.path.join(tmp_dir, zarr_filename)

        # Check if file already exists
        if os.path.exists(zarr_path):
            logger.info(f"Zarr file already exists: {zarr_path}")
            return zarr_path

        S3_CONFIG = {
            "key": ACCESS_KEY_ID,
            "secret": SECRET_ACCESS_KEY,
            "client_kwargs": {
                "endpoint_url": config['endpoint_url'],
                "region_name": "default"
            }
        }
        target_store_config = dict(mode=OpeningMode.CREATE_OVERWRITE)
        convert(AnyPath(product_url, **S3_CONFIG), zarr_path, target_store_kwargs=target_store_config)

        logger.success(f"Downloaded and converted to: {zarr_path}")
        return zarr_path

    except Exception as e:
        logger.error(f"Failed to download Sentinel data: {e}")
        return None

@benchmark
def load_bands_from_s3(s3_client, bucket_name: str, item, bands: list, resize_shape: tuple = (1830, 1830), product_level: str = "L1C") -> np.ndarray:
    """Load bands from S3 storage."""
    if not S3_AVAILABLE:
        logger.error("S3 utilities not available - cannot load bands from S3")
        return None

    try:
        band_data = []
        for band_name in bands:
            try:
                if product_level == "L1C":
                    logger.info("Loading L1C bands from S3 storage")
                    product_url = extract_s3_path_from_url(item.assets[band_name].href)
                else:
                    band_name = f"{band_name}_10m"
                    logger.info("Loading L2A bands from S3 storage")
                    product_url = extract_s3_path_from_url(item.assets[band_name].href)

                content = get_product_content(s3_client, bucket_name, product_url)
                image = Image.open(io.BytesIO(content)).resize(resize_shape)
                band_data.append(np.array(image))
            except Exception as e:
                logger.warning(f"Failed to load band {band_name}: {e}")
                # Create a dummy array if band fails to load
                band_data.append(np.zeros((*resize_shape, 1), dtype=np.uint16))

        logger.success("Loaded bands from S3 storage")
        return np.dstack(band_data)
    except Exception as e:
        logger.error(f"Failed to load bands from S3 storage: {e}")
        return None

def normalize(data_array):
    """
    Normalize the data array to the range [0, 1].
    """
    normalized_data = []
    valid_masks= []
    for i in range(data_array.shape[2]):
        band_data = data_array[:, :, i]
        valid_mask = (band_data > 0)
        result = band_data.copy().astype(np.float32)
        # result[valid_mask] = result[valid_mask] / 10000
        # result[valid_mask] = np.clip(result[valid_mask], 0, 1)
        result[~valid_mask] = 0.0
        normalized_data.append(result)
        valid_masks.append(valid_mask)
    return np.dstack(normalized_data), np.dstack(valid_masks)

@benchmark
def preprocess(zarr_path: str, config: dict, device: torch.device):
    """Preprocess the raw data from Zarr file."""
    try:
        if not os.path.exists(zarr_path):
            logger.error(f"Zarr file does not exist: {zarr_path}")
            return None, None, None

        res_key = f"r{config['res']}"
        x_res = f"x_{config['res']}"
        y_res = f"y_{config['res']}"
        bands = config['bands']

        logger.info(f"Opening Zarr file: {zarr_path}")
        datatree = xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False, chunks={})
        data = datatree.measurements.reflectance[res_key]
        data = data.to_dataset()
        data = data[bands].to_dataarray()

        # Get chunk layout
        chunk_size_y = data.chunksizes[y_res][0]
        chunk_size_x = data.chunksizes[x_res][0]
        nb_chunks_y = len(data.chunksizes[y_res])
        nb_chunks_x = len(data.chunksizes[x_res])

        logger.info(f"Processing {nb_chunks_y}x{nb_chunks_x} chunks of size {chunk_size_y}x{chunk_size_x}")

        all_chunks, all_masks = [], []

        for row in range(nb_chunks_y):  # Y direction
            for col in range(nb_chunks_x):  # X direction
                y_start = row * chunk_size_y
                x_start = col * chunk_size_x
                chunk_ds = data.isel(
                    {y_res: slice(y_start, y_start + chunk_size_y),
                     x_res: slice(x_start, x_start + chunk_size_x)}
                )

                chunk_array = chunk_ds.values.astype(np.float32)
                chunk_array, mask_array = normalize(chunk_array)

                # Convert to torch [C, H, W]
                chunk_tensor = torch.from_numpy(chunk_array).float()
                mask_tensor = torch.from_numpy(mask_array).float()

                # Resize to target size
                chunk_tensor = F.interpolate(
                    chunk_tensor.unsqueeze(0),
                    size=(320, 320),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0),
                    size=(320, 320),
                    mode='nearest'
                ).squeeze(0)
                mask_tensor = mask_tensor > 0.5

                all_chunks.append(chunk_tensor)
                all_masks.append(mask_tensor)

        chunks_grid = torch.stack(all_chunks).view(nb_chunks_y, nb_chunks_x, *all_chunks[0].shape)
        masks_grid = torch.stack(all_masks).view(nb_chunks_y, nb_chunks_x, *all_masks[0].shape)
        meta = (nb_chunks_y, nb_chunks_x, chunk_size_y, chunk_size_x)
        chunks_grid = chunks_grid.unsqueeze(0).to(device)
        masks_grid = masks_grid.unsqueeze(0).to(device)
        datatree.close()
        logger.success("Preprocessed raw data successfully")
        return chunks_grid, masks_grid, meta

    except Exception as e:
        logger.error(f"Failed to preprocess raw data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None


@benchmark
def load_model(model_cfg: dict, weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load the model."""
    try:

        model = define_model(
            name=model_cfg['MODEL']['model_name'],
            encoder_name=model_cfg['MODEL']['encoder_name'],
            encoder_weights=model_cfg['MODEL']['encoder_weights'],
            in_channel=len(model_cfg['DATASET']['bands']),
            out_channels=len(model_cfg['DATASET']['bands']),
            activation=model_cfg['MODEL']['activation']
        )
        model = load_model_weights(model, filename=weights_path)
        logger.success("Model Loaded")
        return model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

@benchmark
def predict_with_metrics_and_bottleneck(model: torch.nn.Module, x_tensor: torch.Tensor, masks_tensor: torch.Tensor, bands: list, device: torch.device) -> tuple:
    """Make a prediction, calculate comprehensive metrics, and capture bottleneck activations and input data."""
    try:
        model.eval()

        # Initialize metrics tracker
        metrics_tracker = MultiSpectralMetrics(bands=bands, device=device)

        # Set up bottleneck capture
        bottleneck_activations = {}

        def get_bottleneck_activation(name):
            def hook(module, input, output):
                bottleneck_activations[name] = output.detach().cpu()
            return hook

        # Register hook for bottleneck layer (adjust layer path as needed)
        # This captures the encoder bottleneck - you may need to adjust the path based on your model architecture
        try:
            hook_handle = model[0].encoder.blocks[5].register_forward_hook(get_bottleneck_activation("bottleneck"))
        except (AttributeError, IndexError):
            # Fallback - try different common bottleneck locations
            try:
                hook_handle = model.encoder.blocks[5].register_forward_hook(get_bottleneck_activation("bottleneck"))
            except (AttributeError, IndexError):
                logger.warning("Could not register bottleneck hook - bottleneck will not be captured")
                hook_handle = None

        with torch.no_grad():
            B, ny, nx, C, H, W = x_tensor.shape
            x_flat = x_tensor.view(B * ny * nx, C, H, W)
            masks_flat = masks_tensor.view(B * ny * nx, C, H, W)

            # Store original input data for saving later
            input_data = x_flat.detach().cpu()

            outputs = model(x_flat)

            # Calculate metrics
            metrics_tracker.reset()
            metrics_tracker.update(outputs, x_flat, masks_flat)
            metrics = metrics_tracker.compute()

            # Clean up hook
            if hook_handle:
                hook_handle.remove()

            logger.success("Prediction, metrics calculation, bottleneck capture, and input data capture successful")
            return outputs, metrics, bottleneck_activations.get("bottleneck", None), input_data

    except Exception as e:
        logger.error(f"Failed to generate prediction with metrics and bottleneck: {e}")
        return None, None, None, None

@benchmark
def save_bottleneck_and_input_data(bottleneck_tensor: torch.Tensor, input_tensor: torch.Tensor, zarr_path: str, output_dir: str, save_format: str = "numpy") -> tuple:
    """
    Save bottleneck matrix and original input data to files.

    Args:
        bottleneck_tensor (torch.Tensor): Bottleneck activations tensor
        input_tensor (torch.Tensor): Original input tensor
        zarr_path (str): Path to the zarr file for metadata
        output_dir (str): Directory to save data
        save_format (str): Format to save ('numpy', 'torch', 'both')

    Returns:
        tuple: (bottleneck_paths, input_paths) - Paths to saved files
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Extract product name from zarr path
        product_name = os.path.basename(zarr_path).replace('.zarr', '').replace('/', '')

        bottleneck_saved_paths = []
        input_saved_paths = []

        # Save bottleneck tensor
        if bottleneck_tensor is not None:
            # Convert to numpy
            bottleneck_np = bottleneck_tensor.cpu().numpy() if torch.is_tensor(bottleneck_tensor) else bottleneck_tensor

            if save_format in ["numpy", "both"]:
                # Save as numpy array (.npy)
                numpy_path = os.path.join(output_dir, f"{product_name}_bottleneck.npy")
                np.save(numpy_path, bottleneck_np)
                bottleneck_saved_paths.append(numpy_path)
                logger.info(f"Bottleneck saved as numpy: {numpy_path}")

            if save_format in ["torch", "both"]:
                # Save as PyTorch tensor (.pt)
                torch_path = os.path.join(output_dir, f"{product_name}_bottleneck.pt")
                torch.save(bottleneck_tensor, torch_path)
                bottleneck_saved_paths.append(torch_path)
                logger.info(f"Bottleneck saved as PyTorch tensor: {torch_path}")

            # Log bottleneck statistics
            logger.info(f"Bottleneck tensor shape: {bottleneck_tensor.shape}")
            logger.info(f"Bottleneck tensor size: {bottleneck_tensor.numel()} elements")
            logger.info(f"Bottleneck memory size: {bottleneck_tensor.numel() * bottleneck_tensor.element_size() / (1024**2):.2f} MB")

            # Compute basic statistics
            flat_bottleneck = bottleneck_np.flatten()
            logger.info(f"Bottleneck statistics: mean={np.mean(flat_bottleneck):.4f}, std={np.std(flat_bottleneck):.4f}")
            logger.info(f"Bottleneck range: min={np.min(flat_bottleneck):.4f}, max={np.max(flat_bottleneck):.4f}")
        else:
            logger.warning("No bottleneck tensor to save")

        # Save input tensor
        if input_tensor is not None:
            # Convert to numpy
            input_np = input_tensor.cpu().numpy() if torch.is_tensor(input_tensor) else input_tensor

            if save_format in ["numpy", "both"]:
                # Save as numpy array (.npy)
                input_numpy_path = os.path.join(output_dir, f"{product_name}_input.npy")
                np.save(input_numpy_path, input_np)
                input_saved_paths.append(input_numpy_path)
                logger.info(f"Input data saved as numpy: {input_numpy_path}")

            if save_format in ["torch", "both"]:
                # Save as PyTorch tensor (.pt)
                input_torch_path = os.path.join(output_dir, f"{product_name}_input.pt")
                torch.save(input_tensor, input_torch_path)
                input_saved_paths.append(input_torch_path)
                logger.info(f"Input data saved as PyTorch tensor: {input_torch_path}")

            # Log input statistics
            logger.info(f"Input tensor shape: {input_tensor.shape}")
            logger.info(f"Input tensor size: {input_tensor.numel()} elements")
            logger.info(f"Input memory size: {input_tensor.numel() * input_tensor.element_size() / (1024**2):.2f} MB")

            # Compute basic statistics
            flat_input = input_np.flatten()
            logger.info(f"Input statistics: mean={np.mean(flat_input):.4f}, std={np.std(flat_input):.4f}")
            logger.info(f"Input range: min={np.min(flat_input):.4f}, max={np.max(flat_input):.4f}")
        else:
            logger.warning("No input tensor to save")

        logger.success(f"Data saved successfully")
        logger.success(f"Bottleneck files: {bottleneck_saved_paths}")
        logger.success(f"Input files: {input_saved_paths}")

        return (bottleneck_saved_paths, input_saved_paths)

    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        return (None, None)

@benchmark
def save_metrics_to_csv(metrics: dict, zarr_path: str, output_dir: str) -> str:
    """
    Save metrics to CSV file with comprehensive information.

    Args:
        metrics (dict): Dictionary with metrics for each band from MultiSpectralMetrics
        zarr_path (str): Path to the zarr file for metadata
        output_dir (str): Directory to save CSV

    Returns:
        str: Path to saved CSV file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Extract product name from zarr path
        product_name = os.path.basename(zarr_path).replace('.zarr', '').replace('/', '')

        # Prepare data for CSV
        csv_data = []

        for band, band_metrics in metrics.items():
            row = {
                'product_name': product_name,
                'band': band,
                'psnr': band_metrics['psnr'],
                'rmse': band_metrics['rmse'],
                'ssim': band_metrics['ssim'],
                'sam': band_metrics['sam']
            }
            csv_data.append(row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f"{product_name}_metrics.csv")
        df.to_csv(csv_path, index=False)

        logger.success(f"Metrics saved to CSV: {csv_path}")
        logger.info(f"Saved metrics for {len(csv_data)} bands")

        # Print summary statistics
        logger.info("Metrics Summary:")
        for metric in ['psnr', 'rmse', 'ssim', 'sam']:
            values = [row[metric] for row in csv_data]
            logger.info(f"  {metric.upper()}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

        return csv_path

    except Exception as e:
        logger.error(f"Failed to save metrics to CSV: {e}")
        return None

@benchmark
def generate_plot_band(gt_np: np.ndarray, pred_np: np.ndarray, bands: list, cmap: str, output_dir: str) -> None:
    """
    Visualize the results with a simple histogram comparison for prediction vs reference.

    Args:
        gt_np: Ground truth data (L2A) with shape [H, W, C]
        pred_np: Predicted data (L2A) array with shape [H, W, C]
        bands: List of band names
        cmap: Colormap to use for visualization
        output_dir: Directory to save the output images
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        for idx, band in enumerate(bands):
            # Create a figure with images and a simple histogram
            fig = plt.figure(figsize=(20, 12))

            # Define a grid layout - 2 rows, 3 columns with bigger image row
            grid = plt.GridSpec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

            # Top row - images
            ax_img1 = plt.subplot(grid[0, 0])
            ax_img2 = plt.subplot(grid[0, 1])
            ax_img3 = plt.subplot(grid[0, 2])

            # Bottom row - just one histogram comparing prediction and reference
            ax_hist = plt.subplot(grid[1, :])

            # Get the data for this band
            band_gt = gt_np[:, :, idx]
            band_pred = pred_np[:, :, idx]

            # Calculate the difference
            diff_target_pred = (np.abs(band_gt - band_pred) / band_gt) * 100

            im1 = ax_img1.imshow(band_gt, cmap=cmap, vmin=0, vmax=1)
            ax_img1.set_title(f"Reference L2A Sen2Cor - Band: {band}", fontsize=14)
            ax_img1.axis('off')
            plt.colorbar(im1, ax=ax_img1, fraction=0.046, pad=0.04)

            im2 = ax_img2.imshow(band_pred, cmap=cmap, vmin=0, vmax=1)
            ax_img2.set_title(f"Prediction L2A - Band: {band}", fontsize=14)
            ax_img2.axis('off')
            plt.colorbar(im2, ax=ax_img2, fraction=0.046, pad=0.04)

            im3 = ax_img3.imshow(diff_target_pred, cmap='plasma', vmin=0, vmax=100)
            ax_img3.set_title(f"Relative Error [%] - {band}", fontsize=14)
            ax_img3.axis('off')
            plt.colorbar(im3, ax=ax_img3, fraction=0.046, pad=0.04)

            # Simple histogram comparison
            # Filter out zeros and NaN values
            gt_data = band_gt[band_gt > 0].flatten()
            pred_data = band_pred[band_pred > 0].flatten()
            # Find common x-axis limits
            min_val = min(gt_data.min(), pred_data.min())
            max_val = max(np.percentile(gt_data, 98), np.percentile(pred_data, 98))

            # Create bins
            bins = np.linspace(min_val, max_val, 100)

            # Plot histograms
            ax_hist.hist(gt_data, bins=bins, alpha=0.5, color='green', label='Reference L2A')
            ax_hist.hist(pred_data, bins=bins, alpha=0.5, color='red', label='Prediction L2A')
            ax_hist.set_title(f"Histogram Comparison - Band {band}", fontsize=14)
            ax_hist.set_xlabel("Pixel Value", fontsize=12)
            ax_hist.set_ylabel("Frequency", fontsize=12)
            ax_hist.legend(fontsize=12)
            ax_hist.set_xlim(0,1)

            # Add metrics
            rmse = np.sqrt(np.mean((band_gt - band_pred)**2))
            mae = np.mean(np.abs(band_gt - band_pred))

            # Add metrics as text to the plot
            ax_hist.text(0.99, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}',
                        transform=ax_hist.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                        fontsize=12)
            # Save the figure
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            fig.savefig(f"{output_dir}/{band}.svg", dpi=300, bbox_inches='tight')
            plt.close(fig)

        logger.success(f"Visualizations with histograms generated in {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())


def plot_benchmark_results(function_durations, output_dir):
    # Convert durations from seconds to minutes
    avg_durations = {k: sum(v) / len(v) / 60 for k, v in function_durations.items()}

    # Avoid log scale issues
    epsilon = 1e-6
    avg_durations = {k: max(val, epsilon) for k, val in avg_durations.items()}

    # Hardcoded Sen2Cor processing time in minutes
    sen2cor_time_min = 35

    plt.figure(figsize=(12, 6))
    plt.bar(avg_durations.keys(), avg_durations.values(), color='skyblue')

    plt.yscale('log')
    plt.ylabel('Duration (minutes, log scale)')
    plt.title('S2 L2A compression/decompression')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_results.svg", dpi=300)
    plt.close()


def reassemble_to_original(outputs_flat, meta):
    """
    Reassemble model outputs from flattened chunks back to original raster dimensions.

    Args:
        outputs_flat (torch.Tensor): Flattened output chunks [N, C, H, W] where N = ny * nx
        meta (tuple): Metadata (nb_chunks_y, nb_chunks_x, chunk_size_y, chunk_size_x)

    Returns:
        torch.Tensor: Full reconstructed image [C, Y, X]
    """
    # Extract metadata
    nb_chunks_y, nb_chunks_x, chunk_size_y, chunk_size_x = meta

    N, C, H, W = outputs_flat.shape

    # Verify that N matches the expected number of chunks
    expected_chunks = nb_chunks_y * nb_chunks_x
    if N != expected_chunks:
        logger.warning(f"Expected {expected_chunks} chunks, got {N}")

    # Calculate full image dimensions
    full_height = nb_chunks_y * chunk_size_y
    full_width = nb_chunks_x * chunk_size_x

    full_image = torch.zeros((C, full_height, full_width))

    # Reassemble chunks
    chunk_idx = 0
    for i in range(nb_chunks_y):
        for j in range(nb_chunks_x):
            if chunk_idx >= N:
                logger.warning(f"Not enough chunks available: {chunk_idx} >= {N}")
                break

            # Resize chunk back to original size
            chunk = F.interpolate(
                outputs_flat[chunk_idx].unsqueeze(0),
                size=(chunk_size_y, chunk_size_x),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            # Place chunk in full image
            y_start, y_end = i * chunk_size_y, (i + 1) * chunk_size_y
            x_start, x_end = j * chunk_size_x, (j + 1) * chunk_size_x
            full_image[:, y_start:y_end, x_start:x_end] = chunk

            chunk_idx += 1

    return full_image


@benchmark
def create_comprehensive_comparison_plot(input_data, output_data, bands, output_dir, sample_idx=0, chunk_idx=0):
    """
    Create comprehensive comparison plots for input vs output - one figure per band.

    Args:
        input_data (torch.Tensor): Input tensor
        output_data (torch.Tensor): Output tensor
        bands (list): List of band names
        output_dir (str): Directory to save the plots
        sample_idx (int): Sample index to visualize
        chunk_idx (int): Chunk index to visualize
    """
    try:
        # Convert to numpy and select specific sample/chunk
        if torch.is_tensor(input_data):
            input_np = input_data[chunk_idx].cpu().numpy()
            output_np = output_data[chunk_idx].cpu().numpy()
        else:
            input_np = input_data[chunk_idx]
            output_np = output_data[chunk_idx]

        os.makedirs(output_dir, exist_ok=True)

        # Create one figure per band
        for idx, band in enumerate(bands):
            fig = plt.figure(figsize=(15, 5))
            gs = GridSpec(1, 3, hspace=0.3, wspace=0.3)

            # Input image
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(input_np[idx], cmap='viridis', vmin=0, vmax=1)
            ax1.set_title(f'Input - {band}', fontsize=14, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # Output image
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(output_np[idx], cmap='viridis', vmin=0, vmax=1)
            ax2.set_title(f'Output - {band}', fontsize=14, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # Difference image
            ax3 = fig.add_subplot(gs[0, 2])
            diff = np.abs(input_np[idx] - output_np[idx])
            im3 = ax3.imshow(diff, cmap='plasma', vmin=0, vmax=diff.max())
            ax3.set_title(f'Abs Difference - {band}', fontsize=14, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # Add metrics text
            rmse = np.sqrt(np.mean((input_np[idx] - output_np[idx])**2))
            mae = np.mean(np.abs(input_np[idx] - output_np[idx]))
            ax3.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}',
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                    fontsize=12)

            plt.suptitle(f'Band Comparison - {band} (Sample {sample_idx}, Chunk {chunk_idx})',
                         fontsize=16, fontweight='bold')

            # Save individual band figure
            band_output_path = os.path.join(output_dir, f'band_comparison_{band}.png')
            plt.tight_layout()
            plt.savefig(band_output_path, dpi=300, bbox_inches='tight')
            plt.close()

        logger.success(f"Band comparison plots saved to {output_dir}")

    except Exception as e:
        logger.error(f"Failed to create comprehensive comparison plots: {e}")


def plot_rgb_composite(reconstruction, bands, output_path="tci_composite.png", figsize=(15, 5)):
    """
    Create RGB composite visualization.

    Args:
        reconstruction (torch.Tensor): Reconstructed image [C, H, W]
        bands (list): List of band names
        output_path (str): Path to save the visualization
        figsize (tuple): Figure size
    """
    try:
        # Find RGB band indices (B04=Red, B03=Green, B02=Blue)
        rgb_indices = []

        for band in ['b04', 'b03', 'b02']:
            if band in bands:
                rgb_indices.append(bands.index(band))
            else:
                logger.warning(f"Band {band} not found for RGB composite")
                return

        if len(rgb_indices) != 3:
            logger.error("Cannot create RGB composite - missing required bands")
            return

        # Extract RGB data
        if torch.is_tensor(reconstruction):
            rgb_data = reconstruction[rgb_indices].detach().cpu().numpy()
        else:
            rgb_data = reconstruction[rgb_indices]

        rgb_image = np.transpose(rgb_data, (1, 2, 0))

        # Normalize for display
        rgb_image = np.clip(rgb_image, 0, 1)

        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # RGB composite
        ax.imshow(rgb_image)
        ax.set_title('RGB Composite (B04-B03-B02)', fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.success(f"RGB composite saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to create RGB composite: {e}")


def main() -> None:
    # Set up logging
    logger.add("log_enhanced_inference.log", rotation="10 MB")
    logger.info("Start enhanced inference workflow with MultiSpectralMetrics...")

    # Load environment and configs
    env = initialize_env()

    dir_path = os.getcwd()

    model_cfg = load_config(f"{dir_path}/weight/config.yaml")
    query_cfg = load_config(f"{dir_path}/cfg/inference_query_config.yaml")
    model_path = f"{dir_path}/weight/EffinientNet-b2-compressor.pth"

    # Setup
    endpoint_url = query_cfg["endpoint_url"]
    bucket_name = query_cfg["bucket_name"]
    stac_url = query_cfg["endpoint_stac"]

    # Only connect to S3 if available
    s3, s3_client = None, None
    catalog = None
    if S3_AVAILABLE:
        s3, s3_client = connect_to_s3(endpoint_url, env["access_key_id"], env["secret_access_key"])
    if PYSTAC_AVAILABLE:
        catalog = pystac_client.Client.open(stac_url)

    # Fetch data
    bands = model_cfg["DATASET"]["bands"]
    bbox = query_cfg["query"]["bbox"]
    start_date = query_cfg["query"]["start_date"]
    end_date = query_cfg["query"]["end_date"]
    max_cloud_cover = query_cfg["query"]["max_cloud_cover"]
    product_url  = data_query(catalog, bbox, start_date, end_date, max_cloud_cover)
    print(product_url)
    zarr_path = download_sentinel_data(env, query_cfg, product_url, dir_path)

    # For demo purposes, use existing zarr file
    # zarr_path = "/mnt/disk/dataset/sentinel-ai-processor/V4/test/target/S2A_MSIL2A_20180319T104021_N0500_R008_T31TFL_20230905T060417.zarr/"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_cfg, model_path, device)

    # Preprocess data
    chunks_grid, masks_grid, meta = preprocess(zarr_path=zarr_path,
                                               config=model_cfg['DATASET'],
                                               device=device)

    # Run prediction with comprehensive metrics and bottleneck capture
    pred, metrics, bottleneck_tensor, input_data = predict_with_metrics_and_bottleneck(model=model,
                                                                                       x_tensor=chunks_grid,
                                                                                       masks_tensor=masks_grid,
                                                                                       bands=bands,
                                                                                       device=device)

    logger.info(f"Prediction shape: {pred.shape}")
    logger.info("Comprehensive metrics computed:")
    for band, band_metrics in metrics.items():
        logger.info(f"  {band}: PSNR={band_metrics['psnr']:.2f} dB, SSIM={band_metrics['ssim']:.4f}, SAM={band_metrics['sam']:.4f} rad, RMSE={band_metrics['rmse']:.5f}")

    # Reconstruct full resolution image from chunks
    logger.info("Reconstructing full resolution image...")
    full_reconstruction = reassemble_to_original(pred.cpu(), meta)
    logger.info(f"Reconstruction shape: {full_reconstruction.shape}")

    # Create output directory for results
    output_dir = os.path.join(dir_path, "inference_results")
    os.makedirs(output_dir, exist_ok=True)

    # Save comprehensive metrics to CSV
    csv_path = save_metrics_to_csv(metrics, zarr_path, output_dir)

    # Save bottleneck matrix and original input data
    bottleneck_paths, input_paths = save_bottleneck_and_input_data(bottleneck_tensor, input_data, zarr_path, output_dir, save_format="both")

    # Create comprehensive comparison plot for element 9
    logger.info("Generating comprehensive comparison plots for chunk element 9...")
    B, ny, nx, C, H, W = chunks_grid.shape
    chunks_flat = chunks_grid.view(B * ny * nx, C, H, W)

    # Check if element 9 exists in the batch
    chunk_element_idx = 9
    if chunk_element_idx < chunks_flat.shape[0]:
        comparison_output_dir = os.path.join(output_dir, "chunk_comparisons")
        create_comprehensive_comparison_plot(
            input_data=chunks_flat.cpu(),
            output_data=pred.cpu(),
            bands=bands,
            output_dir=comparison_output_dir,
            sample_idx=0,
            chunk_idx=chunk_element_idx
        )
        logger.success(f"Comprehensive comparison plots saved for chunk element {chunk_element_idx}")
    else:
        logger.warning(f"Chunk element {chunk_element_idx} does not exist in batch (max index: {chunks_flat.shape[0]-1})")

    # Generate RGB composite visualization from reconstruction
    rgb_output_path = os.path.join(output_dir, "rgb_composite.png")
    plot_rgb_composite(
        reconstruction=full_reconstruction,
        bands=bands,
        output_path=rgb_output_path
    )

    # Plot benchmark results
    logger.info("Generating benchmark results")
    plot_benchmark_results(function_durations=function_durations, output_dir=output_dir)

    logger.success("Enhanced inference pipeline with MultiSpectralMetrics and bottleneck capture completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Metrics CSV: {csv_path}")
    logger.info(f"Bottleneck matrix files: {bottleneck_paths}")
    logger.info(f"Original input data files: {input_paths}")
    logger.info(f"Chunk comparison plots: {output_dir}/chunk_comparisons/")
    logger.info(f"RGB composite: {rgb_output_path}")
    logger.info(f"Benchmark results: {output_dir}/benchmark_results.svg")



if __name__ == "__main__":
    main()