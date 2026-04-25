import numpy as np
from skimage import transform, filters
from edges import get_edges_from_piece

pca_mean = None
pca_components = None

# resizing image data to use for PCA
def resize(patches, n_components=16):
    global pca_mean, pca_components
    
    # flatten each patch and pad to same size due to rotation changes
    flattened_patches = []
    for p in patches:
        flattened = p.ravel().astype(np.float32)
        flattened_patches.append(flattened)
    
    max_size = max(len(p) for p in flattened_patches)

    # pad smaller patches with zeros to match max size
    padded_list = []
    for p in flattened_patches:
        padded_patch = np.pad(p, (0, max_size - len(p)))
        padded_list.append(padded_patch)
    padded = np.array(padded_list, dtype=np.float32)
    
    # computes the mean of each feature across all patches
    pca_mean = np.mean(padded, axis=0)
    X_centered = padded - pca_mean
    
    # decomposes the centered data matrix
    c, v, vt = np.linalg.svd(X_centered, full_matrices=False)
    pca_components = vt[:min(n_components, len(vt))]

# works on a patches PCA
def pca(patch):
    flattened = patch.ravel().astype(np.float32)
    # pad to match training size
    padded = np.pad(flattened, (0, len(pca_mean) - len(flattened)))
    return (padded - pca_mean) @ pca_components.T

# normalized cross-correlation
def ncc(v1, v2):
    # flattening
    v1 = np.asarray(v1, dtype=np.float32).ravel()
    v2 = np.asarray(v2, dtype=np.float32).ravel()
    
    # same size for both vectors
    size = min(v1.size, v2.size)
    v1, v2 = v1[:size], v2[:size]
    
    # stats for each vector
    mean1 = np.mean(v1)
    mean2 = np.mean(v2)
    std1 = np.std(v1)
    std2 = np.std(v2)
    
    # ncc formula returned
    return float(np.clip(np.mean((v1 - mean1) * (v2 - mean2)) / (std1 * std2), -1.0, 1.0))

# store edge stats 
def edge_calculations(patch):
    patch = patch.astype(np.float32)
    
    # gets stats for the patch
    mean = np.mean(patch)
    std = np.std(patch)
    histogram = np.histogram(patch.ravel(), bins=32, range=(0, 256))[0].astype(np.float32)
    
    # pull out the sobel gradients and calculate
    gx = filters.sobel(patch, axis=1)
    gy = filters.sobel(patch, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_strength = np.mean(grad_mag)
    
    # perform pca
    pca_feature = pca(patch)
    
    # return important metrics for the patch
    return {
        'patch': patch,
        'mean': mean,
        'std': std,
        'histogram': histogram,
        'grad_strength': grad_strength,
        'pca_features': pca_feature
    }

# calculate the score with metric comparisons
def compare_metrics(desc1, desc2):
    # NCC
    ncc_score = ncc(desc1['patch'], desc2['patch'])
    
    # statistical similarity - compares mean values of the edge patches
    mean_sim = 1.0 - np.clip(abs(desc1['mean'] - desc2['mean']) / 255.0, 0, 1)
    
    # histogram similarity - compares color/brightness distributions of the edges
    h1 = desc1['histogram'] + 1e-5
    h2 = desc2['histogram'] + 1e-5
    hist_similarity = 1.0 / (1.0 + (0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2))))
    
    # gradient strength similarity - how well edges align
    gradient_similarity = 1.0 - np.clip(abs(desc1['grad_strength'] - desc2['grad_strength']) / 255.0, 0, 1)
    
    # PCA feature similarity - complex patterns in edge patches
    pca_sim = 0.5 
    pca_features_1 = desc1.get('pca_features')
    pca_features_2 = desc2.get('pca_features')
    
    if all([pca_features_1, pca_features_2]):
        pca_correlation = ncc(pca_features_1, pca_features_2)
        pca_sim = 0.5 * (pca_correlation + 1.0)
    
    # return all 5 metrics
    return {
        'ncc': ncc_score,
        'histogram': hist_similarity,
        'pca': pca_sim,
        'gradient': gradient_similarity,
        'mean': mean_sim
    }

# Generate all 4 rotations of an image (0°, 90°, 180°, 270°)
def rotate(img):
    rotations = []
    for rotation_iterations in range(4):
        rotated_image = np.rot90(img, rotation_iterations)
        rotations.append(rotated_image)
    return rotations

# finding the best matches 
def best_match(piece_i, piece_j, i, j, sigma, logger=None):
    best = None
    best_score = -1.0
    
    # pre-compute all rotations for both pieces
    rotations_i = []
    for rotation_index in range(4):
        rotated_piece = np.rot90(piece_i, rotation_index)
        rotations_i.append((rotation_index, rotated_piece))
    
    rotations_j = []
    for rotation_index in range(4):
        rotated_piece = np.rot90(piece_j, rotation_index)
        rotations_j.append((rotation_index, rotated_piece))
    
    # extract edge strips from each rotated piece
    strips_i = []
    for rotation_index, rotated_piece in rotations_i:
        edge_strips = get_edges_from_piece(rotated_piece, strip=30, sigma=sigma, logger=logger)
        strips_i.append((rotation_index, edge_strips))
    
    strips_j = []
    for rotation_index, rotated_piece in rotations_j:
        edge_strips = get_edges_from_piece(rotated_piece, strip=30, sigma=sigma, logger=logger)
        strips_j.append((rotation_index, edge_strips))
    
    # calculate edge details for each strip
    edge_i = []
    for rotation_index, edge_strips in strips_i:
        descriptors = {}
        for edge_name, edge_patch in edge_strips.items():
            descriptors[edge_name] = edge_calculations(edge_patch)
        edge_i.append((rotation_index, descriptors))
    
    edge_j = []
    for rotation_index, edge_strips in strips_j:
        descriptors = {}
        for edge_name, edge_patch in edge_strips.items():
            descriptors[edge_name] = edge_calculations(edge_patch)
        edge_j.append((rotation_index, descriptors))
    
    for rot_i, descs_i in edge_i:
        for rot_j, descs_j in edge_j:
            # right-left match based of best score
            metrics = compare_metrics(descs_i["right"], descs_j["left"])
            score = np.mean([metrics['ncc'], metrics['histogram'], metrics['pca'], metrics['gradient'], metrics['mean']])
            if score > best_score:
                best_score = score
                best = (i, j, "right", rot_i, rot_j, score, metrics)
            
            # bottom-top match based on best score
            metrics = compare_metrics(descs_i["bottom"], descs_j["top"])
            score = np.mean([metrics['ncc'], metrics['histogram'], metrics['pca'], metrics['gradient'], metrics['mean']])
            if score > best_score:
                best_score = score
                best = (i, j, "bottom", rot_i, rot_j, score, metrics)
    
    return best

# matching
def match_pieces(pieces, candidate_pairs, sigma=1.0, epsilon=0.0, logger=None):
    # fit PCA on all edge patches
    all_patches = []
    for piece in pieces:
        strips = get_edges_from_piece(piece, strip=30, sigma=sigma, logger=logger, verbose=True)
        all_patches.extend(strips.values())
    resize(all_patches)
    
    # match piece pairs based on threshold
    matches = []
    for i, j in candidate_pairs:
        best_match = best_match(pieces[i], pieces[j], i, j, sigma, logger=logger)
        if best_match and best_match[5] > epsilon:
            matches.append(best_match)
    
    # sort matches by score 
    matches = sorted(matches, key=lambda x: x[5], reverse=True)
    
    return matches

# reconstruction of the grid
def reconstruct_grid(matches, num_pieces, rows, cols):
    # build match dictionary
    match_dict = {} 
    for m in matches:
        i, j, direction,rot_i,rot_j, score = m[:6]
        key = (i, direction)
        if key not in match_dict:
            match_dict[key] = []
        match_dict[key].append((j,rot_j, score))
    
    # sorted matches
    for key in match_dict:
        match_dict[key].sort(key=lambda x: -x[2])
    
    # initialize grid with first piece at 0,0
    grid = [[None] * cols for _ in range(rows)]
    grid[0][0] = (0, 0)
    selected = {0}
    
    # add to the grid and complete it 
    changed = True
    while changed:
        changed = False
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] is None:
                    continue
                
                piece = grid[r][c]
                
                # fill right neighbor matches
                if c + 1 < cols and grid[r][c+1] is None:
                    key = (piece, "right")
                    if key in match_dict:
                        neighbor, neighbor_rotation, _ = match_dict[key][0]
                        if neighbor not in selected:
                            grid[r][c+1] = (neighbor, neighbor_rotation)
                            selected.add(neighbor)
                            changed = True

                # fill bottom neighbor matches
                if r + 1 < rows and grid[r+1][c] is None:
                    key = (piece, "bottom")
                    if key in match_dict:
                        neighbor, neighbor_rotation, _ = match_dict[key][0]
                        if neighbor not in selected:
                            grid[r+1][c] = (neighbor, neighbor_rotation)
                            selected.add(neighbor)
                            changed = True
    
    # fill remaining grid with unselected pieces
    remaining = [p for p in range(num_pieces) if p not in selected]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None and remaining:
                grid[r][c] = (remaining.pop(0), 0)
    
    return grid