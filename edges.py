import cv2
import numpy as np

# --- Preprocessing ---
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    return blur

# --- Edge Detection ---
def get_edges_and_contours(img):
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return edges, contours

# --- Shape Feature Extraction ---
def extract_edge_features(contour):
    contour = contour.squeeze()
    if len(contour.shape) < 2:
        return None
    return contour.flatten()

# --- Curvature Classification (simple proxy) ---
def classify_edge(contour):
    # Approximate curvature using arc length vs area
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    if area == 0:
        return "flat"
    
    ratio = perimeter / (area + 1e-5)
    
    if ratio > 0.5:
        return "convex"
    else:
        return "concave"

# --- Generate Candidate Matches ---
def generate_candidate_pairs(contours_list):
    candidates = []
    
    for i in range(len(contours_list)):
        for j in range(i+1, len(contours_list)):
            
            for c1 in contours_list[i]:
                for c2 in contours_list[j]:
                    
                    type1 = classify_edge(c1)
                    type2 = classify_edge(c2)
                    
                    # Match concave with convex
                    if (type1 == "concave" and type2 == "convex") or \
                       (type1 == "convex" and type2 == "concave"):
                        
                        candidates.append((i, j))
    
    return list(set(candidates))

# --- Evaluation (Edge Accuracy) ---
def edge_accuracy(pred_matches, gt_matches):
    correct = len(set(pred_matches) & set(gt_matches))
    return correct / len(gt_matches)
