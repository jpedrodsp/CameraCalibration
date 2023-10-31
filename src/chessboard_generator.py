import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
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
    size = (8,8)
    factor = 10
    resolution = (size[0] * factor, size[1] * factor)
    generate_chessboard(file='result/chessboard.png', size=(8,8), resolution=resolution)