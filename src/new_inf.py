import io
import os
import sys
import time
import warnings
from functools import wraps
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

# Project imports
from model_zoo.models import define_model
from utils.torch import load_model_weights
from utils.utils import load_config

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
def predict(model: torch.nn.Module, x_tensor: torch.Tensor) -> np.ndarray:
    """Make a prediction."""
    try:
        model.eval()
        with torch.no_grad():

            B, ny, nx, C, H, W = x_tensor.shape
            x_tensor = x_tensor.view(B * ny * nx, C, H, W)
            outputs = model(x_tensor)
        logger.success("Prediction successfull")
        return outputs

    except Exception as e:
        logger.error(f"Failed to generate L2A: {e}")
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

@benchmark
def generate_tci_plot(gt_np: np.ndarray, pred_np: np.ndarray, bands: list, output_dir: str) -> None:
    """
    Generate True Color Image (RGB composite) plots for both input and predicted data.

    Args:
        gt_np: Input data array with shape [H, W, C]
        gt_np: Ground true L2A data array with shape [H, W, C]
        pred_np: Predicted data array with shape [H, W, C]
        bands: List of band names
        output_dir: Directory to save the output images
    """
    try:
        # Find indices for RGB bands (B04-Red, B03-Green, B02-Blue)
        rgb_indices = []
        for rgb_band in bands:
            if rgb_band in bands:
                rgb_indices.append(bands.index(rgb_band))
            else:
                logger.error(f"Required band {rgb_band} not found in the available bands")
                return

        if len(rgb_indices) != 3:
            logger.error("Could not find all required RGB bands")
            return
        rgb_indices = rgb_indices[::-1]
        # Extract RGB bands
        rgb_x = gt_np[:, :, rgb_indices].copy()  # Make a copy to avoid modifying the original data
        rgb_pred = pred_np[:, :, rgb_indices].copy()
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # Plot gt  TCI
        axs[0].imshow(rgb_x)
        axs[0].set_title(f"Reference L2A Sen2Cor- True Color Index {bands}", fontsize=16)
        axs[0].axis('off')

        # Plot predicted TCI
        axs[1].imshow(rgb_pred)
        axs[1].set_title(f"Predicted L2A - True Color Index {bands}", fontsize=16)
        axs[1].axis('off')

        # Save figure
        fig.tight_layout()
        fig.savefig(f"{output_dir}/TCI.svg", dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.success("TCI RGB composite visualization generated")
    except Exception as e:
        logger.error(f"Failed to generate TCI RGB composite: {e}")


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
    plt.title('Tile Generation Benchmark (Log Scale)')
    plt.xticks(rotation=45, ha='right')

    # Add horizontal line for Sen2Cor processing time
    plt.axhline(y=sen2cor_time_min, color='red', linestyle='--', label='Sen2Cor processing time (35 min)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_results.svg", dpi=300)
    plt.close()


def main() -> None:
    # Set up logging
    logger.add("log_comporessor.log", rotation="10 MB")
    logger.info("Start workflow ...")
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
    s3, s3_client = connect_to_s3(endpoint_url, env["access_key_id"], env["secret_access_key"])
    catalog = pystac_client.Client.open(stac_url)

    # # Fetch data
    bands = model_cfg["DATASET"]["bands"]
    bbox = query_cfg["query"]["bbox"]
    start_date = query_cfg["query"]["start_date"]
    end_date = query_cfg["query"]["end_date"]
    max_cloud_cover = query_cfg["query"]["max_cloud_cover"]

    product_url  = data_query(catalog, bbox, start_date, end_date, max_cloud_cover)

    zarr_path = download_sentinel_data(env, query_cfg, product_url, dir_path)
    # zarr_path = "/home/ubuntu/project/sentinel-2-ai-compressor/src/tmp/S2B_MSIL2A_20240607T102559_N0510_R108_T32ULV_20240607T133635.zarr"
    print(zarr_path)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_cfg, model_path, device)

    chunks_grid, masks_grid, meta = preprocess(zarr_path=zarr_path,
                                               config=model_cfg['DATASET'],
                                               device=device)

    pred = predict(model=model, x_tensor=chunks_grid)
    logger.info(pred.shape)
    plt.imshow(pred[1,10,:,:].detach().cpu().numpy())
    plt.savefig("test.png")


    #TODO

    # # Inference
    # resize = model_cfg["TRAINING"]["resize"]
    # x_tensor, valid_mask = preprocess(raw_data=l2a_raw_data, resize=resize, device=device)
    # pred_np = predict(model=model, x_tensor=x_tensor)

    # x_np, pred_np = postprocess(x_tensor=x_tensor, pred_tensor=pred_np, valid_mask=valid_mask)

    # # Visualization
    # generate_plot_band(gt_np=x_np, pred_np=pred_np, bands=bands, cmap="Grays_r", output_dir=dir_path)
    # generate_tci_plot(gt_np=x_np, pred_np=pred_np, bands=bands[::-1], output_dir=dir_path)


    # logger.info("Plot tile generation benchmark")

    # plot_benchmark_results(function_durations=function_durations, output_dir=dir_path)

    # logger.success("Workflow completed")



if __name__ == "__main__":
    main()
