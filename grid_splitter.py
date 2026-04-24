from skimage import io
import numpy as np
import os
import random
import shutil

def split_grid(image_path, rows, cols, output_folder):
    # start fresh folders
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    # read in the files and split into pieces
    img = io.imread(image_path)
    h, w, _ = img.shape
    
    # determine the height and width of each piece
    piece_height = h // rows
    piece_width = w // cols

    # save pieces into a output folder of pieces
    os.makedirs(output_folder, exist_ok=True)
    
    pieces = []
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            # extract each piece using the heights and width determined earlier
            piece = img[i * piece_height:(i + 1) * piece_height, j * piece_width:(j + 1) * piece_width]

            # randomly rotate pieces (0, 90, 180, 270)
            rotation = random.randint(0, 3)
            piece = np.rot90(piece, rotation)
            
            # save the pieces into a folder
            io.imsave(f"{output_folder}/{count}.png", piece)
            pieces.append(piece)
            count += 1
    
    return pieces

# split the brutus image
img = "brutus_truth.png"
output_folder = "brutus_puzzle_pieces"
rowsb, colsb = 3, 3
split_grid(img, rowsb, colsb, output_folder)

# split the japan image
img2 = "japan_truth.png"
output_folder = "japan_puzzle_pieces"
rowsj, colsj = 3, 3
split_grid(img2, rowsj, colsj, output_folder)

# split the cookies image
img2 = "cookies_truth.png"
output_folder = "cookies_puzzle_pieces"
rowsc, colsc = 2, 2
split_grid(img2, rowsc, colsc, output_folder)