import os
import numpy as np
import shutil
import glob
from skimage import io, transform
from edges import get_edges, generate_candidates
from matches import match_pieces, reconstruct_grid
from grid_splitter import split_grid
from validate import validate_reconstruction

# instead of console logging - saved into a file
class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w')
    
    def log(self, message=""):
        self.file.write(message + "\n")
        self.file.flush()
    
    def close(self):
        self.file.close()

# load pieces from the folder to sort
def load_pieces(folder):
    pieces = []
    for file in sorted(os.listdir(folder)):
        path = os.path.join(folder, file)
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img = io.imread(path)
            if img is not None:
                pieces.append(img)
    return pieces

# connects the pieces together for reconstruction
def build_image(grid, pieces):
    rows, cols = len(grid), len(grid[0])
    h, w = pieces[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            piece_id, rotation = grid[r][c]
            piece = np.rot90(pieces[piece_id], rotation)
            
            # dimension fix up from rotations
            if piece.shape[:2] != (h, w):
                piece = transform.resize(piece, (h, w), order=1, mode='edge')
                if piece.ndim == 2:
                    piece = np.stack([piece] * 3, axis=2)
                piece = (piece * 255).astype(np.uint8)
            
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = piece
    
    return canvas

# main pipeline
def pipeline(folder, rows, cols, sigma=2.0, epsilon=0.5, previous_matches=None, iteration=1, logger=None):
    # get pieces
    pieces = load_pieces(folder)

    # start logging
    if logger is None:
        logger = Logger.__dict__.get('log', print)
    
    log = logger.log if hasattr(logger, 'log') else print

    log(f"Image: {folder}")
    log(f"Parameters: sigma={sigma}, epsilon={epsilon}")

    # edge extraction & candidate pairs
    candidate_pairs = generate_candidates(len(pieces))

    # matching 
    matches = match_pieces(pieces, candidate_pairs, sigma=sigma, epsilon=epsilon, logger=logger)

    # merge with prev. matches - keep better scores
    if previous_matches:
        match_dict = {}
        # new matches
        for m in matches:
            i, j, direction, ri, rj, score = m[:6]
            key = (i, j, direction)
            if key not in match_dict or score > match_dict[key][5]:
                match_dict[key] = m
        # previous matches
        for m in previous_matches:
            i, j, direction, ri, rj, score = m[:6]
            key = (i, j, direction)
            if key not in match_dict or score > match_dict[key][5]:
                match_dict[key] = m
        matches = sorted(match_dict.values(), key=lambda x: -x[5])

    # logging
    log("\nMatches Dictionary:")
    for m in matches[:10]:
        i, j, direction, ri, rj, score = m[:6]
        metrics = m[6] if len(m) > 6 else {}
        if metrics:
            log(f"{i} → {j} ({direction}) | rot_i={ri}, rot_j={rj} | {score:.4f}")
            log(f"  METRICS - NCC: {metrics['ncc']:.4f}, Hist: {metrics['histogram']:.4f}, " +
                f"PCA: {metrics['pca']:.4f}, Grad: {metrics['gradient']:.4f}, Mean: {metrics['mean']:.4f}")
        else:
            log(f"{i} → {j} ({direction}) | rot_i={ri}, rot_j={rj} | {score:.4f}")

    # grid reconstruction
    grid = reconstruct_grid(matches, len(pieces), rows, cols)
    result = build_image(grid, pieces)

    # validate against ground truth image
    puzzle_name = folder.replace("_puzzle_pieces", "")
    truth_path = f"{puzzle_name}_truth.png"
    validation = validate_reconstruction(result, truth_path, grid, pieces, puzzle_name, iteration, logger)

    # save output image
    puzzle_name = folder.replace("_puzzle_pieces", "")
    output_filename = f"reconstructed_{puzzle_name}_iter{iteration}.png"
    io.imsave(output_filename, result)

    return grid, matches, result, validation

# main

# create results directory for logs
results_dir = "iteration_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# clean up old files
for file in glob.glob("reconstructed_*.png"):
    os.remove(file)

puzzle_folders = ["brutus_puzzle_pieces", "japan_puzzle_pieces", "cookies_puzzle_pieces"]
for folder in puzzle_folders:
    if os.path.exists(folder):
        shutil.rmtree(folder)

# generate new randomized puzzle pieces
split_grid("brutus_truth.png", 3, 3, "brutus_puzzle_pieces")
split_grid("japan_truth.png", 3, 3, "japan_puzzle_pieces")
split_grid("cookies_truth.png", 3, 3, "cookies_puzzle_pieces")

# dictionary to store matches and images from each iteration
all_matches = {
    "brutus_puzzle_pieces": [],
    "japan_puzzle_pieces": [],
    "cookies_puzzle_pieces": []
}

# storage
all_images = {
    "brutus": [],
    "japan": [],
    "cookies": []
}

# diff sigma and epsilon values
iterations = [
    (1.0, 0.2),
    (2.0, 0.5),
    (3.0, 0.8),
    (2.0, 0.3),
    (1.5, 0.5),
]

for iteration_num, (sigma, epsilon) in enumerate(iterations, 1):
    log_filename = os.path.join(results_dir, f"iteration_{iteration_num}_results.txt")
    logger = Logger(log_filename)
    
    logger.log(f"ITERATION {iteration_num}: sigma={sigma}, epsilon={epsilon}")
    logger.log("")
    
    puzzles = [
        ("brutus_puzzle_pieces", 3, 3, "brutus"),
        ("japan_puzzle_pieces", 3, 3, "japan"),
        ("cookies_puzzle_pieces", 3, 3, "cookies"),
    ]

    # run pipeline for each puzzle
    for folder, rows, cols, puzzle_name in puzzles:
        grid, matches, img, val = pipeline(folder, rows, cols, sigma=sigma, epsilon=epsilon, previous_matches=all_matches[folder], iteration=iteration_num, logger=logger)
        all_matches[folder] = matches
        all_images[puzzle_name].append(img)

    logger.close()
    
