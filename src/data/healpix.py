

def get_bands(data_tree, res):

    res_key = f"r{res}"
    data = data_tree.measurements.reflectance[res_key]
    return list(data.keys())

def get_chunk_info(data_tree, band, res):
    """
    Extract chunk size and number of chunks from a dataset.

    Parameters:
    - data_tree: xarray.DataTree
    - band: str, e.g. "b03"
    - resolution: str, y-dimension name (e.g. "y_10m")
    - x_res: str, x-dimension name (e.g. "x_10m")

    Returns:
    - chunk_size_y: int
    - chunk_size_x: int
    - nb_chunks_y: int
    - nb_chunks_x: int
    """
    res_key = f"r{res}"
    y_res = f"y_{res}"
    x_res = f"x_{res}"
    data_tree = data_tree.measurements.reflectance[res_key]

    chunk_size_y = data_tree[band].chunksizes[y_res][0]
    chunk_size_x = data_tree[band].chunksizes[x_res][0]
    nb_chunks_y = len(data_tree[band].chunksizes[y_res])
    nb_chunks_x = len(data_tree[band].chunksizes[x_res])


    return chunk_size_y, chunk_size_x, nb_chunks_y, nb_chunks_x

def get_chunk(data_tree, res, chunk_y_idx, chunk_x_idx, chunk_size_y, chunk_size_x):
    """
    Extract a specific chunk from a given band at a given spatial resolution in a DataTree.

    Parameters:
    - data_tree: xarray.DataTree
        The root DataTree object loaded from a Zarr store (e.g., xr.open_datatree(...)).
    - band: str
        The band name to extract (e.g., "b03").
    - res: str
        The spatial resolution as a string (e.g., "10m", "20m", "60m").
    - chunk_y_idx: int
        Index of the chunk along the vertical (y) axis.
    - chunk_x_idx: int
        Index of the chunk along the horizontal (x) axis.

    Returns:
    - xarray.DataArray
        A DataArray corresponding to the specified chunk.
    """
    res_key = f"r{res}"
    y_res = f"y_{res}"
    x_res = f"x_{res}"
    data = data_tree.measurements.reflectance[res_key]

    y_start = chunk_y_idx * chunk_size_y
    x_start = chunk_x_idx * chunk_size_x
    return data.isel(
        {
         x_res: slice(x_start, x_start + chunk_size_x),
         y_res: slice(y_start, y_start + chunk_size_y),}
    )