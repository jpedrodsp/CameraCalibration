import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, argparse
from typing import Tuple

def generate_chessboard(file: str, size: Tuple[int, int], resolution: Tuple[int, int] = (640, 480)):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    # Create a chessboard
    chessboard = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    chessboard[1::2, ::2] = 255
    chessboard[::2, 1::2] = 255
    
    # Expand the chessboard to target resolution
    chessboard = cv2.resize(chessboard, resolution, interpolation=cv2.INTER_NEAREST)
    
    # Save the chessboard
    cv2.imwrite(file, chessboard)
    print('Chessboard saved to "{}"'.format(file))
    plt.axis('off')                 # Hide the axes
    
    # Show the chessboard
    plt.imshow(chessboard)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a chessboard image')
    parser.add_argument('--file', type=str, default='result/chessboard.png', help='Path to the output file')
    parser.add_argument('--width', type=int, default=8, help='Number of squares in the width')
    parser.add_argument('--height', type=int, default=8, help='Number of squares in the height')
    parser.add_argument('--factor', type=int, default=46, help='Size of each square in pixels')
    args = parser.parse_args()
    
    filename = args.file
    size = args.width, args.height
    factor = args.factor
    resolution = (size[0] * factor, size[1] * factor)
    generate_chessboard(file=filename, size=size, resolution=resolution)