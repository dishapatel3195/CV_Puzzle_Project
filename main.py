import os
import cv2
from edges import preprocess, edging, approximate_contours, filter_weak_edges, generate_candidate_pairs
from matches import match_pieces, assemble

def load_pieces(folder):
    pieces = []
    piece_paths = []
    for file in sorted(os.listdir(folder)):
        path = os.path.join(folder, file)
        if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        img = cv2.imread(path)
        if img is not None:
            pieces.append(img)
            piece_paths.append(path)
    return pieces, piece_paths

def run_pipeline(folder):
    pieces, piece_paths = load_pieces(folder)

    piece_ids = list(range(len(pieces)))
    contours_list = []
    for path in piece_paths:
        _, blur = preprocess(path)
        edges, _ = edging(blur)
        contours = approximate_contours(edges)
        contours = filter_weak_edges(contours, min_length=10)
        contours_list.append(contours)
    
    # Teammate 1 output
    candidate_meta = generate_candidate_pairs(contours_list, piece_ids)
    candidate_pairs = list({tuple(sorted((i, j))) for i, j, *_ in candidate_meta})
    
    # Teammate 2 processing
    matches = match_pieces(pieces, candidate_pairs)
    assembly = assemble(matches)
    
    return assembly

if __name__ == "__main__":
    result = run_pipeline("brutus_puzzle_pieces")
    print(result)