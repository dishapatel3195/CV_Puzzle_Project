import numpy as np
from skimage import io
from skimage import feature
from skimage.color import rgb2gray
from scipy import ndimage


# grayscale image
def preprocess(img):
    if isinstance(img, str):
        img_array = io.imread(img)
    else:
        img_array = img

    gray = rgb2gray(img_array)
    gray = (gray * 255).astype(np.uint8)

    return gray


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
    for i in range(num_pieces):
        for j in range(num_pieces):
            if i != j:
                pairs.append((i, j))
    return pairs


def get_edges_from_piece(img, strip=30, sigma=1.0, use_gradients=False):

    from scipy.ndimage import gaussian_filter
    
    h, w = img.shape[:2]
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        gray = img.astype(np.uint8)
    
    # Apply Gaussian blur with adjustable sigma for noise reduction
    blurred = gaussian_filter(gray.astype(np.float32), sigma=sigma)
    
    if use_gradients:
        # Use Sobel gradients for edge detection
        gx = ndimage.sobel(blurred, axis=1)
        gy = ndimage.sobel(blurred, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        # Normalize to 0-1
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-5)
        # Blend with blurred image
        blurred = 0.6 * blurred / 255.0 + 0.4 * magnitude
    else:
        blurred = blurred / 255.0
    
    # Convert to 0-255 range
    blurred = (blurred * 255).astype(np.uint8)
    
    # Extract edge strips
    return {
        "top": enhance_contrast(blurred[0:strip, :]),
        "bottom": enhance_contrast(blurred[h-strip:h, :]),
        "left": enhance_contrast(blurred[:, 0:strip]),
        "right": enhance_contrast(blurred[:, w-strip:w]),
    }


def enhance_contrast(patch):
    """Enhance local contrast in a patch"""
    patch_float = patch.astype(np.float32)
    mean = patch_float.mean()
    std = patch_float.std()
    if std > 0:
        patch_float = (patch_float - mean) / std * 30 + 128
        patch_float = np.clip(patch_float, 0, 255)
    return patch_float.astype(np.uint8)