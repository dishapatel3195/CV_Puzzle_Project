import os
import numpy as np
import shutil
import glob
from datetime import datetime
from skimage import io, transform
from edges import get_edges, generate_candidates
from matches import match_pieces, reconstruct_grid
from grid_splitter import split_grid
from validate import validate_reconstruction


class Logger:
    """Logs output to a file only"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w')
    
    def log(self, message=""):
        self.file.write(message + "\n")
        self.file.flush()
    
    def close(self):
        self.file.close()


def load_pieces(folder):
    pieces = []
    for file in sorted(os.listdir(folder)):
        path = os.path.join(folder, file)
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img = io.imread(path)
            if img is not None:
                pieces.append(img)
    return pieces

def build_image(grid, pieces):
    """Assemble puzzle from grid configuration"""
    rows, cols = len(grid), len(grid[0])
    h, w = pieces[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            piece_id, rotation = grid[r][c]
            piece = np.rot90(pieces[piece_id], rotation)
            
            # Handle dimension mismatch from rotation
            if piece.shape[:2] != (h, w):
                piece = transform.resize(piece, (h, w), order=1, mode='edge')
                if piece.ndim == 2:
                    piece = np.stack([piece] * 3, axis=2)
                piece = (piece * 255).astype(np.uint8)
            
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = piece
    
    return canvas

def run_pipeline(folder, rows, cols, sigma=2.0, epsilon=0.5, use_gradients=True, previous_matches=None, iteration=1, logger=None):
    pieces = load_pieces(folder)

    if logger is None:
        logger = Logger.__dict__.get('log', print)
    
    log = logger.log if hasattr(logger, 'log') else print

    log(f"Processing: {folder}")
    log(f"Parameters: sigma={sigma}, epsilon={epsilon}, gradients={use_gradients}")

    # Step 1: Edge extraction (debug/optional)
    edges_list = [get_edges(p) for p in pieces]

    # Step 2: Candidate pairs
    candidate_pairs = generate_candidates(len(pieces))

    # Step 3: Matching with tunable parameters
    matches = match_pieces(pieces, candidate_pairs, sigma=sigma, epsilon=epsilon, use_gradients=use_gradients)

    # Step 3b: Merge with previous matches if available (keep better scores)
    if previous_matches:
        log(f"  Merging with {len(previous_matches)} previous matches...")
        match_dict = {}
        # Add new matches
        for m in matches:
            i, j, direction, ri, rj, score = m[:6]
            key = (i, j, direction)
            if key not in match_dict or score > match_dict[key][5]:
                match_dict[key] = m
        # Add previous matches if not superseded
        for m in previous_matches:
            i, j, direction, ri, rj, score = m[:6]
            key = (i, j, direction)
            if key not in match_dict or score > match_dict[key][5]:
                match_dict[key] = m
        matches = sorted(match_dict.values(), key=lambda x: -x[5])
        log(f"  Total unique matches after merge: {len(matches)}")

    log("\nTop Matches:")
    for m in matches[:10]:
        i, j, direction, ri, rj, score = m[:6]
        metrics = m[6] if len(m) > 6 else {}
        if metrics:
            log(f"{i} → {j} ({direction}) | rot_i={ri}, rot_j={rj} | {score:.4f}")
            log(f"  Metrics - NCC: {metrics['ncc']:.4f}, Hist: {metrics['histogram']:.4f}, " +
                f"PCA: {metrics['pca']:.4f}, Grad: {metrics['gradient']:.4f}, Mean: {metrics['mean']:.4f}")
        else:
            log(f"{i} → {j} ({direction}) | rot_i={ri}, rot_j={rj} | {score:.4f}")

    # Step 4: Grid reconstruction
    grid = reconstruct_grid(matches, len(pieces), rows, cols)

    # Step 5: Build final image
    result = build_image(grid, pieces)
    
    # Step 6: Validate against ground truth
    puzzle_name = folder.replace("_puzzle_pieces", "")
    truth_path = f"{puzzle_name}_truth.png"
    validation = validate_reconstruction(result, truth_path, grid, pieces, puzzle_name, iteration, logger)
    
    # Step 7: Save output image
    puzzle_name = folder.replace("_puzzle_pieces", "")
    output_filename = f"reconstructed_{puzzle_name}_iter{iteration}.png"
    io.imsave(output_filename, result)

    log(f"\nSaved: {output_filename}")

    return grid, matches, result, validation

# main

# Create results directory for logs
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

# Generate new randomized puzzle pieces
split_grid("brutus_truth.png", 3, 3, "brutus_puzzle_pieces")
split_grid("japan_truth.png", 3, 3, "japan_puzzle_pieces")
split_grid("cookies_truth.png", 3, 3, "cookies_puzzle_pieces")

# Dictionary to store matches and images from each iteration
all_matches = {
    "brutus_puzzle_pieces": [],
    "japan_puzzle_pieces": [],
    "cookies_puzzle_pieces": []
}

all_images = {
    "brutus": [],
    "japan": [],
    "cookies": []
}

iterations = [
    # Test 1: Low sigma, low epsilon, no gradients
    (1.0, 0.2, {"brutus": False, "japan": False, "cookies": False}),
    
    # Test 2: Medium sigma, medium epsilon, japan with gradients
    (2.0, 0.5, {"brutus": False, "japan": True, "cookies": False}),
    
    # Test 3: Higher sigma, higher epsilon, all with gradients
    (3.0, 0.8, {"brutus": True, "japan": True, "cookies": True}),
    
    # Test 4: Medium sigma, low epsilon, selective gradients
    (2.0, 0.3, {"brutus": True, "japan": False, "cookies": False}),
    
    # Test 5: Lower sigma, medium epsilon, all with gradients (refinement)
    (1.5, 0.5, {"brutus": True, "japan": True, "cookies": True}),
]

for iteration_num, (sigma, epsilon, gradient_map) in enumerate(iterations, 1):
    # Create logger for this iteration
    log_filename = os.path.join(results_dir, f"iteration_{iteration_num}_results.txt")
    logger = Logger(log_filename)
    
    logger.log(f"ITERATION {iteration_num}: sigma={sigma}, epsilon={epsilon}")
    logger.log(f"Gradient Map: {gradient_map}")
    logger.log("")
    
    print("\n" + "="*60)
    print(f"ITERATION {iteration_num}: sigma={sigma}, epsilon={epsilon}")
    print("="*60)
    
    puzzles = [
        ("brutus_puzzle_pieces", 3, 3, "brutus"),
        ("japan_puzzle_pieces", 3, 3, "japan"),
        ("cookies_puzzle_pieces", 3, 3, "cookies"),
    ]
    
    for folder, rows, cols, puzzle_name in puzzles:
        use_grad = gradient_map[puzzle_name]
        grid, matches, img, val = run_pipeline(folder, rows, cols, sigma=sigma, epsilon=epsilon, use_gradients=use_grad, previous_matches=all_matches[folder], iteration=iteration_num, logger=logger)
        all_matches[folder] = matches
        all_images[puzzle_name].append(img)

    logger.close()
    
