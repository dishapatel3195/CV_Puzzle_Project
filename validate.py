import warnings
import numpy as np
from skimage import io, transform, metrics

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ratio of correctly matched edges to total possible edges
def compute_edge_matching_accuracy(grid, pieces, truth):
    rows, cols = len(grid), len(grid[0])
    total_possible_edges = (rows - 1) * cols + rows * (cols - 1)
    correct_matches = 0
    
    piece_size = pieces[0].shape[0]
    
    # iterate through the grid
    for r in range(rows):
        for c in range(cols):
            piece_id, rotation = grid[r][c]
            
            # look at bottom neighbor and right neighbor
            if c + 1 < cols:
                neighbor_id, neighbor_rot = grid[r][c + 1]
                right_edge = np.rot90(pieces[piece_id], rotation)[-piece_size//4:, :]
                left_edge = np.rot90(pieces[neighbor_id], neighbor_rot)[:, :piece_size//4]
                
                if right_edge.size > 0 and left_edge.size > 0:
                    # flatten
                    right_flat = right_edge.ravel()[:100]
                    left_flat = left_edge.ravel()[:100]
                    if len(right_flat) > 1 and len(left_flat) > 1 and np.std(right_flat) > 0 and np.std(left_flat) > 0:
                        edge_corr = np.corrcoef(right_flat, left_flat)[0, 1]
                        if not np.isnan(edge_corr) and edge_corr > 0.5:
                            correct_matches += 1
            
            if r + 1 < rows:
                neighbor_id, neighbor_rot = grid[r + 1][c]
                bottom_edge = np.rot90(pieces[piece_id], rotation)[-piece_size//4:, :]
                top_edge = np.rot90(pieces[neighbor_id], neighbor_rot)[:piece_size//4, :]
                
                if bottom_edge.size > 0 and top_edge.size > 0:
                    bottom_flat = bottom_edge.ravel()[:100]
                    top_flat = top_edge.ravel()[:100]
                    if len(bottom_flat) > 1 and len(top_flat) > 1 and np.std(bottom_flat) > 0 and np.std(top_flat) > 0:
                        edge_corr = np.corrcoef(bottom_flat, top_flat)[0, 1]
                        if not np.isnan(edge_corr) and edge_corr > 0.5:
                            correct_matches += 1
    
    # ratio
    edge_match_accuracy = correct_matches / max(total_possible_edges, 1)
    return edge_match_accuracy


# ratio of correctly placed pieces to total pieces
def compute_piece_placement_accuracy(reconstructed, truth, grid, pieces):
    rows, cols = len(grid), len(grid[0])
    piece_size = pieces[0].shape[0]
    correct_placements = 0
    total_pieces = rows * cols
    
    for r in range(rows):
        for c in range(cols):
            piece_id, rotation = grid[r][c]
            
            recon_region = reconstructed[r*piece_size:(r+1)*piece_size, c*piece_size:(c+1)*piece_size]
            truth_region = truth[r*piece_size:(r+1)*piece_size, c*piece_size:(c+1)*piece_size]
            
            if recon_region.shape == truth_region.shape:
                recon_flat = recon_region.ravel().astype(float)
                truth_flat = truth_region.ravel().astype(float)
                
                # Normalize only if std > 0
                recon_std = np.std(recon_flat)
                truth_std = np.std(truth_flat)
                
                if recon_std > 0 and truth_std > 0:
                    recon_flat = (recon_flat - np.mean(recon_flat)) / recon_std
                    truth_flat = (truth_flat - np.mean(truth_flat)) / truth_std
                    
                    region_corr = np.corrcoef(recon_flat, truth_flat)[0, 1]
                    
                    if not np.isnan(region_corr) and region_corr > 0.6:
                        correct_placements += 1
    
    piece_placement_accuracy = correct_placements / max(total_pieces, 1)
    return piece_placement_accuracy

def validate_reconstruction(reconstructed, truth_image_path, grid=None, pieces=None, puzzle_name=None, iteration=None, logger=None):
    truth = io.imread(truth_image_path)

    # Resize if needed to match
    if reconstructed.shape != truth.shape:
        reconstructed = transform.resize(reconstructed, truth.shape, order=1, mode='edge')
        reconstructed = (reconstructed * 255).astype(np.uint8)
    
    # 1. MSE (Mean Squared Error)
    mse = np.mean((reconstructed.astype(float) - truth.astype(float)) ** 2)
    
    # 2. Correlation
    recon_norm = reconstructed.astype(float) / 255.0
    truth_norm = truth.astype(float) / 255.0
    correlation = np.corrcoef(recon_norm.ravel(), truth_norm.ravel())[0, 1]
    
    # 3. Histogram correlation per channel
    hist_scores = []
    for i in range(3):
        hist_recon = np.histogram(reconstructed[:,:,i].ravel(), bins=256, range=(0, 256))[0].astype(float)
        hist_truth = np.histogram(truth[:,:,i].ravel(), bins=256, range=(0, 256))[0].astype(float)
        hist_recon = hist_recon / (np.sum(hist_recon) + 1e-5)
        hist_truth = hist_truth / (np.sum(hist_truth) + 1e-5)
        score = np.corrcoef(hist_recon, hist_truth)[0, 1]
        if not np.isnan(score):
            hist_scores.append(score)
    avg_hist_score = np.mean(hist_scores) if hist_scores else 0.0
    
    # 4. Edge Matching Accuracy
    edge_match_accuracy = compute_edge_matching_accuracy(grid, pieces, truth) if grid and pieces else 0.0
    
    # 5. Piece Placement Accuracy
    piece_placement_accuracy = compute_piece_placement_accuracy(reconstructed, truth, grid, pieces) if grid and pieces else 0.0
    
    results = {
        'mse': mse,
        'correlation': correlation,
        'hist_score': avg_hist_score,
        'edge_match_accuracy': edge_match_accuracy,
        'piece_placement_accuracy': piece_placement_accuracy,
        'reconstructed': reconstructed,
        'truth': truth
    }
    
    # Log results if puzzle_name, iteration, and logger provided
    if puzzle_name and iteration and logger:
        log = logger.log if hasattr(logger, 'log') else print
        log(f"{puzzle_name.upper()} - Iteration {iteration} Validation")
        log(f"\n")
        log(f"MSE:              {mse:.2f}")
        log(f"Correlation:     {correlation:.4f}")
        log(f"Histogram Match: {avg_hist_score:.4f}")
        log(f"\n\nEdge Match Accuracy Ratio: {edge_match_accuracy:.2%}")
        log(f"Piece Placement Accuracy Ratio: {piece_placement_accuracy:.2%}")

    return results
