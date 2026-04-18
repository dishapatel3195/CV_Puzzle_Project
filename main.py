import os
import cv2
from edges import preprocess, get_edges_and_contours, generate_candidate_pairs
from matching import match_pieces, assemble


def load_pieces(folder):
    pieces = []
    for file in sorted(os.listdir(folder)):
        piece = cv2.imread(os.path.join(folder, file))
        pieces.append(piece)
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

brutus_output_folder = "brutus_puzzle_pieces"
brutus_assembly = run_pipeline(brutus_output_folder)
japan_output_folder = "japan_puzzle_pieces"
japan_assembly = run_pipeline(japan_output_folder)
cookies_output_folder = "cookies_puzzle_pieces"
cookies_assembly = run_pipeline(cookies_output_folder)
