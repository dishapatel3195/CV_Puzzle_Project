import os
import cv2
from edges import preprocess, get_edges_and_contours, generate_candidate_pairs
from matches import match_pieces, assemble

def load_pieces(folder):
    pieces = []
    for file in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, file))
        pieces.append(img)
    return pieces

def run_pipeline(folder):
    pieces = load_pieces(folder)
    
    processed = [preprocess(p) for p in pieces]
    
    contours_list = []
    for img in processed:
        _, contours = get_edges_and_contours(img)
        contours_list.append(contours)
    
    # Teammate 1 output
    candidate_pairs = generate_candidate_pairs(contours_list)
    
    # Teammate 2 processing
    matches = match_pieces(pieces, candidate_pairs)
    assembly = assemble(matches)
    
    return assembly

if __name__ == "__main__":
    result = run_pipeline("puzzle_pieces")
    print(result)