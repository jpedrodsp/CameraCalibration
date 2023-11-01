SCALEDOWN_FACTOR = 1

def get_image_folder(name: str):
    return f'images/{name}'

def get_image_types():
    return ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']

def get_chessboard_size():
    return (8, 8)

def get_chessboard_frame_size():
    chessboard_size = get_chessboard_size()
    factor = 46
    return (chessboard_size[0] * factor, chessboard_size[1] * factor)