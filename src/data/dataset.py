import torch
import numpy as np
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
from data.transform import get_transforms
import natsort
import xarray as xr
import glob
from PIL import Image
import os
from loguru import logger



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

# def normalize(data_array):
#     """
#     Normalize the data array to the range [0, 1].
#     """
#     normalized_data = []
#     valid_masks= []
#     for i in range(data_array.shape[2]):
#         band_data = data_array[:, :, i]
#         valid_mask = (band_data > 0)
#         valid_pixels = band_data[valid_mask]
#         min_val = np.min(valid_pixels)
#         max_val = np.max(valid_pixels)
#         #lower = np.percentile(valid_pixels, lower_percent)
#         #upper = np.percentile(valid_pixels, upper_percent)
#         # result[valid_mask] = np.clip((band[valid_mask] - lower) / (upper - lower), 0, 1)

#         result = band_data.copy().astype(np.float32)
#         result[valid_mask] = (valid_pixels - min_val) / (max_val - min_val)
#         # result[valid_mask] = result[valid_mask] / 10000
#         result[~valid_mask] = 0.0
#         normalized_data.append(result)
#         valid_masks.append(valid_mask)
#     return np.dstack(normalized_data), np.dstack(valid_masks)


def read_images(product_paths):
    images = []
    for path in product_paths:
        data = Image.open(path)
        data = np.array(data)
        images.append(data)

    # image : - > H x W x C
    images = np.dstack(images)
    return images


class Sentinel2Dataset(Dataset):

    def __init__(self, df_x, df_y, train, augmentation, img_size):
        self.df_x = df_x
        self.df_y = df_y
        self.train = train
        self.augmentation = augmentation
        self.img_size = img_size
        # self.transform = get_transforms(train=self.train, augmentation=True, aug_prob=0.5)

    def __getitem__(self, index):
        x_paths = natsort.natsorted(glob.glob(os.path.join(self.df_y["path"][index], "*.png"), recursive=False))
        x_data = read_images(x_paths)
        x_data, x_mask = normalize(x_data)

        y_paths = natsort.natsorted(glob.glob(os.path.join(self.df_y["path"][index], "*.png"), recursive=False))
        y_data = read_images(y_paths)
        y_data, y_mask = normalize(y_data)

        # Apply the same augmentation to both input and target
        # if self.train and self.augmentation:
        #     transformed = self.transform(image=x_data, mask=y_data)
        #     x_data = transformed["image"]
        #     y_data = transformed["mask"]

        # Handle resizing separately from augmentations
        x_data = cv2.resize(x_data, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        y_data = cv2.resize(y_data, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Resize masks to match image size
        x_mask = cv2.resize(x_mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        y_mask = cv2.resize(y_mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(bool)

        # Final valid mask is intersection of x and y
        valid_mask = torch.from_numpy(y_mask).bool()
        valid_mask = torch.permute(valid_mask, (2, 0, 1))  # HWC to CHW

        x_data = torch.from_numpy(x_data).float()
        x_data = torch.permute(x_data, (2, 0, 1))  # HWC to CHW

        y_data = torch.from_numpy(y_data).float()
        y_data = torch.permute(y_data, (2, 0, 1))  # HWC to CHW

        return x_data, y_data, valid_mask

    def __len__(self):
        return len(self.df_x)

    # def __init__(self, df_x, df_y, train, augmentation, img_size):
    #     self.df_x = df_x
    #     self.df_y = df_y
    #     self.train = train
    #     self.augmentation = augmentation
    #     self.img_size = img_size
    #     # self.transform = get_transforms(train=self.train, augmentation=self.augmentation)

    # def __getitem__(self, index):
    #     x_paths = natsort.natsorted(glob.glob(os.path.join(self.df_x["path"][index], "*.png"), recursive=False))
    #     x_data = read_images(x_paths)
    #     x_data, x_mask = normalize(x_data)
    #     x_data = cv2.resize(x_data, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
    #     x_mask = cv2.resize(x_mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(bool)

    #     y_paths = natsort.natsorted(glob.glob(os.path.join(self.df_y["path"][index], "*.png"), recursive=False))
    #     y_data = read_images(y_paths)
    #     y_data, y_mask  = normalize(y_data)
    #     y_data = cv2.resize(y_data, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
    #     y_mask = cv2.resize(y_mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(bool)

    #     # Final valid mask is intersection of x and y
    #     valid_mask = torch.from_numpy(y_mask).bool()
    #     valid_mask = torch.permute(valid_mask, (2, 0, 1))  # HWC to CHW

    #     x_data = torch.from_numpy(x_data).float()
    #     x_data = torch.permute(x_data, (2, 0, 1))  # HWC to CHW

    #     y_data = torch.from_numpy(y_data).float()
    #     y_data = torch.permute(y_data, (2, 0, 1))  # HWC to CHW

    #     # transformed = self.transform(image=x_data, mask=y_data)
    #     # y_data = transformed["mask"]
    #     # x_data = transformed["image"]

    #     return x_data, y_data, valid_mask

    # def __len__(self):
    #     return len(self.df_x)


class Sentinel2TCIDataset(Dataset):
    def __init__(self, df_path,
                 train,
                 augmentation,
                 img_size):

        self.df_path = df_path
        self.train = train
        self.augmentation = augmentation
        self.img_size = img_size
        self.transform = get_transforms(train=self.train,
                                        augmentation=self.augmentation)

    def __getitem__(self, index):
        # Load images
        x_path = self.df_path.l1c_path.iloc[index]
        x_data = cv2.imread(x_path)
        x_data = cv2.cvtColor(x_data, cv2.COLOR_BGR2RGB)
        x_data = cv2.resize(x_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x_data = np.array(x_data).astype(np.float32) / 255.0
        x_data = torch.from_numpy(x_data).float()
        x_data = torch.permute(x_data, (2, 0, 1))  # HWC to CHW

        y_path = self.df_path.l2a_path.iloc[index]
        y_data = cv2.imread(y_path)
        y_data = cv2.cvtColor(y_data, cv2.COLOR_BGR2RGB)
        y_data = cv2.resize(y_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        y_data = np.array(y_data).astype(np.float32) / 255.0
        y_data = torch.from_numpy(y_data).float()
        y_data = torch.permute(y_data, (2, 0, 1))  # HWC to CHW

        # transformed = self.transform(image=x_data, mask=y_data)
        # y_data = transformed["mask"]
        # x_data = transformed["image"]


        return x_data, y_data

    def __len__(self):
        return len(self.df_path)


# ---------------- Dataset ----------------
# class Sentinel2ZarrDataset(Dataset):
#     def __init__(self, df_x, res, bands, target_size=(320, 320)):
#         self.df_x = df_x
#         self.res = res
#         self.bands = bands
#         self.target_size = target_size
#         self.res_key = f"r{res}"
#         self.x_res = f"x_{res}"
#         self.y_res = f"y_{res}"

#         logger.info(f"Dataset initialized with {len(df_x)} items | res={res} | bands={bands} | target_size={target_size}")

#     def __len__(self):
#         return len(self.df_x)

#     def __getitem__(self, index):
#         zarr_path = self.df_x["path"].iloc[index] + ".zarr"
#         logger.debug(f"[{index}] Opening Zarr file: {zarr_path}")

#         datatree = xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False, chunks={})
#         data = datatree.measurements.reflectance[self.res_key]
#         logger.debug(f"[{index}] Opened reflectance group: {self.res_key}")

#         # Get chunk layout
#         band = self.bands[0]
#         chunk_size_y = data[band].chunksizes[self.y_res][0]
#         chunk_size_x = data[band].chunksizes[self.x_res][0]
#         nb_chunks_y = len(data[band].chunksizes[self.y_res])
#         nb_chunks_x = len(data[band].chunksizes[self.x_res])

#         logger.debug(f"[{index}] Chunk layout: {nb_chunks_y}x{nb_chunks_x}, chunk_size=({chunk_size_y},{chunk_size_x})")

#         all_chunks, all_masks = [], []

#         for row in range(nb_chunks_y):
#             for col in range(nb_chunks_x):
#                 y_start = row * chunk_size_y
#                 x_start = col * chunk_size_x
#                 logger.debug(f"[{index}] Processing chunk ({row},{col}) slice=({y_start}:{y_start+chunk_size_y}, {x_start}:{x_start+chunk_size_x})")

#                 chunk_ds = data.isel(
#                     {self.y_res: slice(y_start, y_start + chunk_size_y),
#                      self.x_res: slice(x_start, x_start + chunk_size_x)}
#                 )


#                 chunk_ds = data[self.bands][y_start: y_start + chunk_size_y,x_start: x_start + chunk_size_x ]


#                 chunk_ds

#                 chunk_array = chunk_ds.to_dataset().to_dataarray().compute().values
#                 # logger.debug(f"[{index}] Chunk ({row},{col}) raw shape: {chunk_array.shape}")

#                 chunk_array, mask_array = normalize(chunk_array)
#                 # logger.debug(f"[{index}] Chunk ({row},{col}) normalized")

#                 # Convert to torch [C, H, W]
#                 chunk_tensor = torch.from_numpy(chunk_array).float()
#                 mask_tensor = torch.from_numpy(mask_array).float()

#                 # Resize to target size
#                 chunk_tensor = F.interpolate(
#                     chunk_tensor.unsqueeze(0),
#                     size=self.target_size,
#                     mode='bilinear',
#                     align_corners=False
#                 ).squeeze(0)

#                 mask_tensor = F.interpolate(
#                     mask_tensor.unsqueeze(0),
#                     size=self.target_size,
#                     mode='nearest'
#                 ).squeeze(0)
#                 mask_tensor = mask_tensor > 0.5

#                 # logger.debug(f"[{index}] Chunk ({row},{col}) resized to {self.target_size}, tensor shape: {chunk_tensor.shape}")

#                 all_chunks.append(chunk_tensor)
#                 all_masks.append(mask_tensor)

#         chunks_grid = torch.stack(all_chunks).view(nb_chunks_y, nb_chunks_x, *all_chunks[0].shape)
#         masks_grid = torch.stack(all_masks).view(nb_chunks_y, nb_chunks_x, *all_masks[0].shape)
#         meta = (nb_chunks_y, nb_chunks_x, chunk_size_y, chunk_size_x)

#         datatree.close()
#         # logger.debug(f"[{index}] Finished processing -> chunks_grid: {chunks_grid.shape}, masks_grid: {masks_grid.shape}")

#         return chunks_grid, masks_grid, meta


class Sentinel2ZarrDataset(Dataset):
    def __init__(self, df_x, res, bands, target_size=(320, 320)):
        self.df_x = df_x
        self.res = res
        self.bands = bands
        self.target_size = target_size
        self.res_key = f"r{res}"
        self.x_res = f"x_{res}"
        self.y_res = f"y_{res}"

    def __len__(self):
        return len(self.df_x)

    def __getitem__(self, index):
        zarr_path = self.df_x["path"].iloc[index] + ".zarr"
        datatree = xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False, chunks={})
        data = datatree.measurements.reflectance[self.res_key]
        data = data.to_dataset()
        data = data[self.bands].to_dataarray()

        # --- Get chunk layout ---
        band  = self.bands[0]
        chunk_size_y = data.chunksizes[self.y_res][0]
        chunk_size_x = data.chunksizes[self.x_res][0]
        nb_chunks_y = len(data.chunksizes[self.y_res])
        nb_chunks_x = len(data.chunksizes[self.x_res])

        all_chunks, all_masks = [], []

        for row in range(nb_chunks_y):  # Y direction
            for col in range(nb_chunks_x):  # X direction
                y_start = row * chunk_size_y
                x_start = col * chunk_size_x
                chunk_ds = data.isel(
                            {self.y_res: slice(y_start, y_start + chunk_size_y),
                            self.x_res: slice(x_start, x_start + chunk_size_x)}
                        )

                chunk_array = chunk_ds.values.astype(np.float32)
                chunk_array, mask_array = normalize(chunk_array)
                # logger.debug(f"[{index}] Chunk ({row},{col}) normalized")

                # Convert to torch [C, H, W]
                chunk_tensor = torch.from_numpy(chunk_array).float()
                mask_tensor = torch.from_numpy(mask_array).float()

                # Resize to target size
                chunk_tensor = F.interpolate(
                    chunk_tensor.unsqueeze(0),
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0),
                    size=self.target_size,
                    mode='nearest'
                ).squeeze(0)
                mask_tensor = mask_tensor > 0.5

                all_chunks.append(chunk_tensor)
                all_masks.append(mask_tensor)

        chunks_grid = torch.stack(all_chunks).view(nb_chunks_y, nb_chunks_x, *all_chunks[0].shape)
        masks_grid = torch.stack(all_masks).view(nb_chunks_y, nb_chunks_x, *all_masks[0].shape)
        meta = (nb_chunks_y, nb_chunks_x, chunk_size_y, chunk_size_x)

        datatree.close()
        # logger.debug(f"[{index}] Finished processing -> chunks_grid: {chunks_grid.shape}, masks_grid: {masks_grid.shape}")

        return chunks_grid, masks_grid, meta