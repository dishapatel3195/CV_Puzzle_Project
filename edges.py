import numpy as np
from skimage import io
from skimage import feature
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy import ndimage


# grayscale image
def preprocess(img):
    img_array = io.imread(img)

    gray = rgb2gray(img_array)
    gray = (gray * 255).astype(np.uint8)

    return gray

# edge extraction using Canny edge detection
def get_edges(img, strip_w=20):
    gray = preprocess(img)
    
    # apply Canny edge detection
    edges_canny = feature.canny(gray, sigma=1.0)
    edges_canny = (edges_canny * 255).astype(np.uint8)

    h, w = edges_canny.shape

    edges = {
        "top": edges_canny[0:strip_w, :],
        "bottom": edges_canny[h-strip_w:h, :],
        "left": edges_canny[:, 0:strip_w],
        "right": edges_canny[:, w-strip_w:w],
    }

    return edges

# pairs to compare edges
def generate_candidates(num_pieces):
    pairs = []
    
    # loop through all the pieces and make pairs to compare
    for i in range(num_pieces):
        for j in range(num_pieces):
            if i != j:
                pairs.append((i, j))
    return pairs

# get edge strips of a piece
def get_edges_from_piece(img, strip=30, sigma=1.0, logger=None, verbose=False):    
    h, w = img.shape[:2]
    
    gray = preprocess(img)
    
    # apply Gaussian blur with adjustable sigma for noise reduction
    blurred = gaussian_filter(gray.astype(np.float32), sigma=sigma)
    
    # use Sobel gradients for edge detection
    gx = ndimage.sobel(blurred, axis=1)
    gy = ndimage.sobel(blurred, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # normalize
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-5)
    blurred = blurred / 255.0 * magnitude
    
    # convert to 0-255 range
    blurred = (blurred * 255).astype(np.uint8)
    
    # log values
    log_func = logger.log if hasattr(logger, 'log') else print
    log_func(f"Sobel Gradients:")
    log_func(f"  Gx range: [{gx.min():.2f}, {gx.max():.2f}], mean: {gx.mean():.2f}")
    log_func(f"  Gy range: [{gy.min():.2f}, {gy.max():.2f}], mean: {gy.mean():.2f}")
    log_func(f"  Magnitude range: [{magnitude.min():.2f}, {magnitude.max():.2f}], mean: {magnitude.mean():.2f}")
    log_func(f"  Edge Strip Statistics (strip_width={strip}):")
    log_func(f"    Image size: {h}×{w}")
    log_func(f"    Blurred value range: [{blurred.min()}, {blurred.max()}], mean: {blurred.mean():.2f}")
    
    edge_strips = {
        "top": blurred[0:strip, :],
        "bottom": blurred[h-strip:h, :],
        "left": blurred[:, 0:strip],
        "right": blurred[:, w-strip:w],
    }
    
    for direction, strip_data in edge_strips.items():
        log_func(f"      {direction.upper():6} strip: shape={strip_data.shape}, mean={strip_data.mean():.2f}, std={strip_data.std():.2f}")
    
    return edge_strips