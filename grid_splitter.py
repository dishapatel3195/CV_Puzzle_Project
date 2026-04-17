import cv2
import numpy as np
import os
import random

def split_grid(image_path, rows, cols, output_folder):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    piece_h = h // rows
    piece_w = w // cols
    
    os.makedirs(output_folder, exist_ok=True)
    
    pieces = []
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            piece = img[i*piece_h:(i+1)*piece_h, j*piece_w:(j+1)*piece_w]
            cv2.imwrite(f"{output_folder}/{count}.png", piece)
            pieces.append(piece)
            count += 1
    
    return pieces


img = "brutus_truth.png"
output_folder = "brutus_puzzle_pieces"
rows, cols = 4, 4
split_grid(img, rows, cols, output_folder)

img2 = "japan_truth.png"
output_folder = "japan_puzzle_pieces"
rows, cols = 4, 4
split_grid(img2, rows, cols, output_folder)

img2 = "cookies_truth.png"
output_folder = "cookies_puzzle_pieces"
rows, cols = 4, 4
split_grid(img2, rows, cols, output_folder)