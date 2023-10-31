import os, glob
import cv2 as cv
import numpy as np
from dataclasses import dataclass

def make_imagegrid(folder: str, output_filepath: str):
    images = glob.glob(folder + '/*.jpg') + glob.glob(folder + '/*.JPG') + glob.glob(folder + '/*.png') + glob.glob(folder + '/*.PNG') + glob.glob(folder + '/*.jpeg') + glob.glob(folder + '/*.JPEG')
    images_count = len(images)
    if images_count == 0:
        raise ValueError('No images found in folder "{}"'.format(folder))
    columns_per_row = 5
    rows_per_page = images_count / columns_per_row
    if rows_per_page % 1.0 != 0.0:
        rows_per_page = int(rows_per_page) + 1
    else:
        rows_per_page = int(rows_per_page)

    dir = os.path.dirname(os.path.abspath(output_filepath))
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    canvas = np.zeros((rows_per_page * 480, columns_per_row * 640, 3), dtype=np.uint8)
    for i, image in enumerate(images):
        print('Processing image {} of {}'.format(i+1, images_count))
        row = i // columns_per_row
        column = i % columns_per_row
        img = cv.imread(image)
        img = cv.resize(img, (640, 480), interpolation=cv.INTER_NEAREST)
        cv.putText(img, str(i+1), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        try:
            canvas[row*480:(row+1)*480, column*640:(column+1)*640] = img
        except:
            print('Error processing image {}'.format(image))
    cv.imwrite(output_filepath, canvas)

@dataclass
class RenderItem:
    folder_path: str
    output_file: str

if __name__ == '__main__':
    item1 = RenderItem("images/camera1_motox4-traseira", "result/motox4_output_grid.jpg")
    make_imagegrid(item1.folder_path, item1.output_file)
    item2 = RenderItem("images/camera2_nintendo-dsi-traseira", "result/nintendo-dsi_output_grid.jpg")
    make_imagegrid(item2.folder_path, item2.output_file)