import cv2
import numpy as np
from sklearn.decomposition import PCA
from itertools import combinations

# --- PCA ---
def apply_pca(features, n_components=10):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    return reduced, pca

# --- Texture Matching ---
def match_texture(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    
    i1 = cv2.resize(img1, (w,h))
    i2 = cv2.resize(img2, (w,h))
    
    result = cv2.matchTemplate(i1, i2, cv2.TM_CCOEFF_NORMED)
    return np.max(result)

# --- Rotations ---
def generate_rotations(img):
    return [np.rot90(img, k) for k in range(4)]

# --- Match Pieces ---
def match_pieces(pieces, candidate_pairs):
    matches = []
    
    for i, j in candidate_pairs:
        best_score = -1
        
        for r1 in generate_rotations(pieces[i]):
            for r2 in generate_rotations(pieces[j]):
                
                score = match_texture(r1, r2)
                
                if score > best_score:
                    best_score = score
        
        matches.append((i, j, best_score))
    
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
    correct = sum([1 for i in pred_positions if pred_positions[i] == gt_positions[i]])
    return correct / len(gt_positions)