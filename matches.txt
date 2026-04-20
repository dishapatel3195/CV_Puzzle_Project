import cv2
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # SVD-based PCA: X_centered = U S Vt
        _, singular_vals, vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components_ = vt[: self.n_components]
        n_samples = X.shape[0]
        denom = max(n_samples - 1, 1)
        self.explained_variance_ = (singular_vals[: self.n_components] ** 2) / denom
        return self

    def transform(self, X):
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA must be fitted before transform")

        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# --- PCA ---
def apply_pca(features, n_components=10):
    features = np.asarray(features, dtype=np.float64)

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array")

    max_components = min(features.shape[0], features.shape[1])
    n_components = max(1, min(n_components, max_components))

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    return reduced, pca

# --- Texture Matching ---
def _to_grayscale(img):
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.ndim == 3:
        if img.shape[2] >= 3:
            b = img[..., 0].astype(np.float32)
            g = img[..., 1].astype(np.float32)
            r = img[..., 2].astype(np.float32)
            return 0.114 * b + 0.587 * g + 0.299 * r
        return img[..., 0].astype(np.float32)
    raise ValueError("Unsupported image shape")


def _resize_nn(img, h, w):
    if img.shape[0] == h and img.shape[1] == w:
        return img
    y_idx = np.linspace(0, img.shape[0] - 1, h).astype(np.int64)
    x_idx = np.linspace(0, img.shape[1] - 1, w).astype(np.int64)
    return img[np.ix_(y_idx, x_idx)]


def _normalized_correlation(i1, i2):
    a = i1.astype(np.float32).ravel()
    b = i2.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def match_texture(img1, img2):
    if img1 is None or img2 is None:
        return -1.0

    if img1.size == 0 or img2.size == 0:
        return -1.0

    gray1 = _to_grayscale(img1)
    gray2 = _to_grayscale(img2)

    h = min(gray1.shape[0], gray2.shape[0])
    w = min(gray1.shape[1], gray2.shape[1])

    if h < 3 or w < 3:
        return -1.0
    
    i1 = _resize_nn(gray1, h, w)
    i2 = _resize_nn(gray2, h, w)

    # Template matching (NCC)
    ncc = float(np.max(cv2.matchTemplate(i1.astype(np.float32), i2.astype(np.float32), cv2.TM_CCOEFF_NORMED)))

    # Normalized correlation
    corr = _normalized_correlation(i1, i2)

    # Color consistency (mean absolute channel difference)
    if img1.ndim == 2:
        c1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        c1 = img1
    if img2.ndim == 2:
        c2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        c2 = img2

    c1 = cv2.resize(c1, (w, h)).astype(np.float32)
    c2 = cv2.resize(c2, (w, h)).astype(np.float32)
    mad = np.mean(np.abs(c1 - c2)) / 255.0
    color = float(1.0 - np.clip(mad, 0.0, 1.0))

    return 0.55 * ncc + 0.30 * corr + 0.15 * color

# --- Rotations ---
def generate_rotations(img):
    return [np.rot90(img, k) for k in range(4)]

# --- Match Pieces ---
def match_pieces(pieces, candidate_pairs):
    matches = []
    seen_pairs = set()

    if not pieces:
        return matches

    # Build PCA-compressed patch features for each piece rotation.
    feature_index = []
    feature_vectors = []
    for i, piece in enumerate(pieces):
        for r, rot_img in enumerate(generate_rotations(piece)):
            gray = _to_grayscale(rot_img)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            patch = cv2.resize(blur, (28, 28)).astype(np.float32).reshape(-1)
            feature_index.append((i, r))
            feature_vectors.append(patch)

    reduced_features, _ = apply_pca(np.asarray(feature_vectors, dtype=np.float64), n_components=24)
    pca_map = {key: reduced_features[idx] for idx, key in enumerate(feature_index)}
    
    for i, j in candidate_pairs:
        if i == j:
            continue

        if i < 0 or j < 0 or i >= len(pieces) or j >= len(pieces):
            continue

        pair = tuple(sorted((i, j)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        best_score = -1.0
        
        for ri, r1 in enumerate(generate_rotations(pieces[i])):
            for rj, r2 in enumerate(generate_rotations(pieces[j])):
                texture_score = match_texture(r1, r2)

                v1 = pca_map[(i, ri)]
                v2 = pca_map[(j, rj)]
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                pca_score = float(np.dot(v1, v2) / denom) if denom != 0 else 0.0

                score = 0.70 * texture_score + 0.30 * pca_score
                
                if score > best_score:
                    best_score = score
        
        matches.append((i, j, float(best_score)))
    
    return sorted(matches, key=lambda x: -x[2])

# --- Assembly ---
def assemble(matches):
    used = set()
    assembly = []
    
    for i, j, score in matches:
        if i not in used and j not in used:
            assembly.append((i, j))
            used.add(i)
            used.add(j)
    
    return assembly

# --- Reconstruction Accuracy ---
def reconstruction_accuracy(pred_positions, gt_positions):
    if not gt_positions:
        return 0.0

    correct = sum(
        1
        for i in gt_positions
        if i in pred_positions and pred_positions[i] == gt_positions[i]
    )
    return correct / len(gt_positions)