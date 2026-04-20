import numpy as np
from skimage import transform, filters
from edges import get_edges_from_piece

pca_mean = None
pca_components = None

# trains PCA
def resize(patches, n_components=16):
    global pca_mean, pca_components
    
    # Flatten each patch and pad to same size
    flattened_patches = []
    for p in patches:
        flattened = p.ravel().astype(np.float32)
        flattened_patches.append(flattened)
    
    max_size = max(len(p) for p in flattened_patches)

    # Pad smaller patches with zeros to match max size
    padded_list = []
    for p in flattened_patches:
        padded_patch = np.pad(p, (0, max_size - len(p)))
        padded_list.append(padded_patch)
    padded = np.array(padded_list, dtype=np.float32)
    
    pca_mean = np.mean(padded, axis=0)
    X_centered = padded - pca_mean
    
    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
    pca_components = vt[:min(n_components, len(vt))]

# works on sinlge-patch PCA
def pca_transform(patch):
    flattened = patch.ravel().astype(np.float32)
    # Pad to match training size
    padded = np.pad(flattened, (0, len(pca_mean) - len(flattened)))
    return (padded - pca_mean) @ pca_components.T

# normalized cross-correlation
def ncc(v1, v2):
    v1 = np.asarray(v1, dtype=np.float32).ravel()
    v2 = np.asarray(v2, dtype=np.float32).ravel()
    
    if v1.size == 0 or v2.size == 0:
        return 0.0
    
    # Ensure same size
    size = min(v1.size, v2.size)
    v1, v2 = v1[:size], v2[:size]
    
    mean1, mean2 = np.mean(v1), np.mean(v2)
    std1, std2 = np.std(v1), np.std(v2)
    
    if std1 == 0 or std2 == 0:
        return 0.0
    
    return float(np.clip(np.mean((v1 - mean1) * (v2 - mean2)) / (std1 * std2), -1.0, 1.0))

# store edge stats 
def edge_calculations(patch):
    patch = patch.astype(np.float32)
    
    mean = np.mean(patch)
    std = np.std(patch)
    histogram = np.histogram(patch.ravel(), bins=32, range=(0, 256))[0].astype(np.float32)
    
    gx = filters.sobel(patch, axis=1)
    gy = filters.sobel(patch, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_strength = np.mean(grad_mag)
    
    pca_feature = pca_transform(patch)
    
    return {
        'patch': patch,
        'mean': mean,
        'std': std,
        'histogram': histogram,
        'grad_strength': grad_strength,
        'pca_features': pca_feature
    }


def compare_metrics(desc1, desc2):
    # Pixel-level NCC
    ncc_score = ncc(desc1['patch'], desc2['patch'])
    
    # Statistical similarity
    mean_sim = 1.0 - np.clip(abs(desc1['mean'] - desc2['mean']) / 255.0, 0, 1)
    
    # Histogram similarity (chi-squared)
    h1 = desc1['histogram'] + 1e-5
    h2 = desc2['histogram'] + 1e-5
    chi2 = 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2))
    hist_sim = 1.0 / (1.0 + chi2)
    
    # Gradient strength similarity
    grad_sim = 1.0 - np.clip(abs(desc1['grad_strength'] - desc2['grad_strength']) / 255.0, 0, 1)
    
    # PCA feature similarity
    pca_sim = 0.5  # Neutral default
    if desc1.get('pca_features') is not None and desc2.get('pca_features') is not None:
        pca_ncc = ncc(desc1['pca_features'], desc2['pca_features'])
        pca_sim = 0.5 * (pca_ncc + 1.0)  # Normalize to 0-1
    
    return {
        'ncc': ncc_score,
        'histogram': hist_sim,
        'pca': pca_sim,
        'gradient': grad_sim,
        'mean': mean_sim
    }


def rotate(img):
    return [np.rot90(img, k) for k in range(4)]


def match_pieces(pieces, candidate_pairs, sigma=1.0, epsilon=0.0, use_gradients=False):
    # Fit PCA on all edge patches
    all_patches = []
    for piece in pieces:
        strips = get_edges_from_piece(piece, strip=30, sigma=sigma, use_gradients=use_gradients)
        all_patches.extend(strips.values())
    resize(all_patches)
    
    # Match piece pairs
    matches = []
    for i, j in candidate_pairs:
        best_match = find_best_match(pieces[i], pieces[j], i, j, sigma, use_gradients)
        if best_match and best_match[5] > epsilon:
            matches.append(best_match)
    
    matches.sort(key=lambda x: -x[5])  # Sort by score (descending)
    
    return matches


def find_best_match(piece_i, piece_j, i, j, sigma, use_gradients):
    """Find best rotation/direction match between two pieces"""
    best = None
    best_score = -1.0
    
    # Pre-compute all rotations and their edges
    rotations_i = [(ri, np.rot90(piece_i, ri)) for ri in range(4)]
    rotations_j = [(rj, np.rot90(piece_j, rj)) for rj in range(4)]
    
    strips_i_all = [(ri, get_edges_from_piece(rot_i, strip=30, sigma=sigma, use_gradients=use_gradients)) 
                    for ri, rot_i in rotations_i]
    strips_j_all = [(rj, get_edges_from_piece(rot_j, strip=30, sigma=sigma, use_gradients=use_gradients)) 
                    for rj, rot_j in rotations_j]
    
    descs_i_all = [(ri, {k: edge_calculations(v) for k, v in strips.items()}) 
                   for ri, strips in strips_i_all]
    descs_j_all = [(rj, {k: edge_calculations(v) for k, v in strips.items()}) 
                   for rj, strips in strips_j_all]
    
    for ri, descs_i in descs_i_all:
        for rj, descs_j in descs_j_all:
            # RIGHT→LEFT match
            metrics = compare_metrics(descs_i["right"], descs_j["left"])
            score = np.mean([metrics['ncc'], metrics['histogram'], metrics['pca'], metrics['gradient'], metrics['mean']])
            if score > best_score:
                best_score = score
                best = (i, j, "right", ri, rj, score, metrics)
            
            # BOTTOM→TOP match
            metrics = compare_metrics(descs_i["bottom"], descs_j["top"])
            score = np.mean([metrics['ncc'], metrics['histogram'], metrics['pca'], metrics['gradient'], metrics['mean']])
            if score > best_score:
                best_score = score
                best = (i, j, "bottom", ri, rj, score, metrics)
    
    return best


def reconstruct_grid(matches, num_pieces, rows, cols):
    # Build match dictionary
    match_dict = {} 
    for m in matches:
        i, j, direction, ri, rj, score = m[:6]
        key = (i, direction)
        if key not in match_dict:
            match_dict[key] = []
        match_dict[key].append((j, rj, score))
    
    # Sort by score
    for key in match_dict:
        match_dict[key].sort(key=lambda x: -x[2])
    
    # Initialize grid with first piece
    grid = [[None] * cols for _ in range(rows)]
    grid[0][0] = (0, 0)
    used = {0}
    
    # Propagate matches through grid
    changed = True
    while changed:
        changed = False
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] is None:
                    continue
                
                piece, rotation = grid[r][c]
                
                # Try to fill RIGHT neighbor
                if c + 1 < cols and grid[r][c+1] is None:
                    key = (piece, "right")
                    if key in match_dict:
                        neighbor, neighbor_rot, _ = match_dict[key][0]
                        if neighbor not in used:
                            grid[r][c+1] = (neighbor, neighbor_rot)
                            used.add(neighbor)
                            changed = True
                
                # Try to fill BOTTOM neighbor
                if r + 1 < rows and grid[r+1][c] is None:
                    key = (piece, "bottom")
                    if key in match_dict:
                        neighbor, neighbor_rot, _ = match_dict[key][0]
                        if neighbor not in used:
                            grid[r+1][c] = (neighbor, neighbor_rot)
                            used.add(neighbor)
                            changed = True
    
    # Fill remaining cells with unused pieces
    remaining = [p for p in range(num_pieces) if p not in used]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None and remaining:
                grid[r][c] = (remaining.pop(0), 0)
    
    return grid