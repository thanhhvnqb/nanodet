import numpy as np


def space_to_depth_np(x: np.array, block_size: int):
    """
    Re-arranges blocks of spatial data into depth.

    Args:
    x: np.ndarray of shape (height, width, depth)
    block_size: int, the size of the spatial block.

    Returns:
    np.ndarray of shape (height // block_size, width // block_size, depth * (block_size ** 2))
    """
    x = np.asarray(x)
    assert x.ndim >= 3, "Expecting an image with at least 3 dimensions (H, W, C)"
    height, width, depth = x.shape
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            "The height and width of the input tensor must be divisible by block_size."
        )

    new_height = height // block_size
    new_width = width // block_size
    new_depth = depth * (block_size**2)

    # Reshape and transpose to interleave the spatial blocks into depth
    x = x.reshape(new_height, block_size, new_width, block_size, depth)
    x = np.swapaxes(x, 1, 2).reshape(new_height, new_width, new_depth)
    return x


def depth_to_space_np(x, block_size):
    """
    Re-arranges data from depth into blocks of spatial data.

    Args:
    x: np.ndarray of shape (height, width, depth)
    block_size: int, the size of the spatial block.

    Returns:
    np.ndarray of shape (height * block_size, width * block_size, depth // (block_size ** 2))
    """
    x = np.asarray(x)
    assert x.ndim >= 3, "Expecting an image with at least 3 dimensions (H, W, C)"
    height, width, depth = x.shape
    new_depth = depth // (block_size**2)
    if depth % (block_size**2) != 0:
        raise ValueError(
            "The depth of the input tensor must be divisible by (block_size ** 2)."
        )

    # Reshape and transpose to interleave the blocks
    x = x.reshape(height, width, block_size, block_size, new_depth)
    x = np.swapaxes(x, 1, 2).reshape(height * block_size, width * block_size, new_depth)
    return x
