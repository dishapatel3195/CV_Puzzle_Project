import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2gray

def preprocess(img_path):
    img_array = io.imread(img_path)
    
    # Convert to grayscale
    gray = rgb2gray(img_array)
    gray = (gray * 255).astype(np.uint8)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return gray, blur

def edging(img):
    # Canny edge detection
    edges = cv2.Canny(img, 100, 200)
    
    # Extract contour coordinates
    contours = np.argwhere(edges > 0)
    
    return edges, contours

def approximate_contours(edges, epsilon=2.0):
    # Find contours using OpenCV (simplified version)
    contours_cv, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    approximated = []
    for contour in contours_cv:
        # Approximate contour to reduce points
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 2:  # Only keep valid contours
            approximated.append(approx.squeeze())
    
    return approximated

def calculate_curvature(contour):
    if len(contour) < 3:
        return []
    
    curvatures = []
    for i in range(len(contour)):
        # Get three consecutive points
        p1 = contour[(i - 1) % len(contour)]
        p2 = contour[i]
        p3 = contour[(i + 1) % len(contour)]
        
        # Calculate cross product to determine curvature direction
        v1 = p2 - p1
        v2 = p3 - p2
        
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        curvatures.append(cross)
    
    return np.array(curvatures)

def classify_edge(contour):
    if len(contour) < 4:
        return "flat"
    
    # Calculate curvature signature
    curvatures = calculate_curvature(contour)
    
    if len(curvatures) == 0:
        return "flat"
    
    # Analyze curvature distribution
    positive_curvature = np.sum(curvatures > 0)
    negative_curvature = np.sum(curvatures < 0)
    total = len(curvatures)
    
    # If mostly positive curvature -> convex
    if positive_curvature > total * 0.6:
        return "convex"
    # If mostly negative curvature -> concave
    elif negative_curvature > total * 0.6:
        return "concave"
    else:
        return "flat"

def filter_weak_edges(contours, min_length=10):
    filtered = []
    for contour in contours:
        if len(contour) >= min_length:
            filtered.append(contour)
    return filtered

def match_edges_geometrically(contour1, contour2, threshold=0.3):
    if len(contour1) < 2 or len(contour2) < 2:
        return 0.0
    
    # Compare edge lengths
    len1 = np.sum(np.linalg.norm(np.diff(contour1, axis=0), axis=1))
    len2 = np.sum(np.linalg.norm(np.diff(contour2, axis=0), axis=1))
    
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Calculate length ratio
    length_ratio = min(len1, len2) / max(len1, len2)
    
    return length_ratio

def generate_candidate_pairs(piece_contours, piece_ids):
    candidates = []
    edge_classifications = {}
    
    # Classify all edges
    for piece_id, contours in zip(piece_ids, piece_contours):
        edge_classifications[piece_id] = []
        for contour in contours:
            edge_type = classify_edge(contour)
            edge_classifications[piece_id].append({
                'contour': contour,
                'type': edge_type
            })
    
    # Match complementary edges
    for i, piece_i in enumerate(piece_ids):
        for j, piece_j in enumerate(piece_ids):
            if i >= j:
                continue
            
            # Compare edges of piece_i with edges of piece_j
            for edge_i in edge_classifications[piece_i]:
                for edge_j in edge_classifications[piece_j]:
                    type_i = edge_i['type']
                    type_j = edge_j['type']
                    
                    # Match concave with convex
                    if (type_i == "concave" and type_j == "convex") or \
                       (type_i == "convex" and type_j == "concave"):
                        
                        # Check geometric compatibility
                        compatibility = match_edges_geometrically(
                            edge_i['contour'], 
                            edge_j['contour']
                        )
                        
                        if compatibility > 0.5:  # Threshold
                            candidates.append((piece_i, piece_j, type_i, type_j))
    
    return candidates

def edge_matching_accuracy(predicted_matches, ground_truth_matches):
    if len(ground_truth_matches) == 0:
        return 0.0
    
    correct = 0
    for pred in predicted_matches:
        # Normalize tuples for comparison (order doesn't matter)
        pred_normalized = tuple(sorted(pred[:2]))
        for gt in ground_truth_matches:
            gt_normalized = tuple(sorted(gt[:2]))
            if pred_normalized == gt_normalized:
                correct += 1
                break
    
    accuracy = correct / len(ground_truth_matches)
    return accuracy

def analyze_failures(predicted_matches, ground_truth_matches):
    failures = {
        'false_positives': [],
        'false_negatives': [],
        'total_predicted': len(predicted_matches),
        'total_ground_truth': len(ground_truth_matches)
    }
    
    pred_set = set(tuple(sorted(p[:2])) for p in predicted_matches)
    gt_set = set(tuple(sorted(g[:2])) for g in ground_truth_matches)
    
    failures['false_positives'] = list(pred_set - gt_set)
    failures['false_negatives'] = list(gt_set - pred_set)
    
    return failures


