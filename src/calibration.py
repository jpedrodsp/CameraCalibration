# forked from https://github.com/niconielsen32/CameraCalibration

import numpy as np
import cv2 as cv
import glob, pickle
import time, json
import threading, multiprocessing, multiprocessing.pool
import common

objpoints = [] # Define objpoints as a global variable
imgpoints = [] # Define imgpoints as a global variable
mutex = threading.Lock()

def callback_calibrate_camera(returnvalue):
    global objpoints, imgpoints
    imgpath, objp, imgp, diff = returnvalue
    try:
        if not objp is None and not imgp is None:
            print(f'\t - [{imgpath}] Adding {len(objp)} object points and {len(imgp)} image points')
            mutex.acquire()
            objpoints.extend(objp)
            imgpoints.extend(imgp)
            mutex.release()
        else:
            print(f'\t - [{imgpath}] No object points or image points found')
    finally:
        print(f'\t - [{imgpath}] Done in {diff} seconds')

def calibration_worker_process(imgpath: str, chessboardSize: tuple, criteria: tuple, objp: np.ndarray):
        print(f'\t - [{imgpath}] Starting')
        start = time.time()
        img = cv.imread(imgpath)
        # Resize the image 2 times smaller
        factor = 1 / 2
        img = cv.resize(img, (0,0), fx=factor, fy=factor, interpolation=cv.INTER_NEAREST)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            imgpoints.append(corners2)
            objpoints.append(objp)
        end = time.time()
        diff = end - start
        cv.imshow(f'{imgpath}', img)
        cv.waitKey(1000)
        
        return (imgpath, objpoints, imgpoints, diff)


def calibrate_camera(src_name: str):
    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
    print(f"Step 1: Finding chessboard corners for {src_name}")

    chessboardSize = (7,7)   # number of inner corners per a chessboard row and column
    frameSize = (368,368)    # 8x images with 46 pixels per square = 368 pixels
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 15
    objp = objp * size_of_chessboard_squares_mm

    images = []
    for image_type in common.get_image_types():
        images += glob.glob(common.get_image_folder(src_name) + '/*.' + image_type)
                
    # Create a pool of processes. Wait for them to complete.
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.pool.Pool() as pool:
        print(f'\t - Found {len(images)} images. Using {num_processes} processes')
        for imgindex, imgpath in enumerate(images):
            print(f'\t - [{imgindex+1}/{len(images)}] Adding {imgpath}')
            pool.apply_async(calibration_worker_process, args=(imgpath, chessboardSize, criteria, objp), callback=callback_calibrate_camera)
        pool.close()
        pool.join()
        print(f'\t - Done')
    cv.destroyAllWindows()


    ############## CALIBRATION #######################################################
    print(f"Step 2: Calibrating camera {src_name}")

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    pickle.dump((cameraMatrix, dist), open( f"result/{src_name}_calibration.pkl", "wb" ))
    pickle.dump(cameraMatrix, open( f"result/{src_name}_cameraMatrix.pkl", "wb" ))
    pickle.dump(dist, open( f"result/{src_name}_dist.pkl", "wb" ))
    with open(f'result/{src_name}_values.txt', 'w') as f:
        data = {
            'matrix': cameraMatrix.tolist(),
        }
        f.write(json.dumps(data))
        


    ############## UNDISTORTION #####################################################
    print(f"Step 3: Undistorting images for {src_name}")

    img = cv.imread('result/chessboard.png')
    h, w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


    # Undistort
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(f'result/{src_name}_caliResult1.png', dst)



    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(f'result/{src_name}_caliResult2.png', dst)




    # Reprojection Error
    print(f"Step 4: Calculating reprojection error for {src_name}")
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    totalerror = mean_error/len(objpoints)
    print( "total error: {}".format(totalerror) )
    return totalerror

if __name__ == '__main__':
    content = ['camera1_motox4-traseira', 'camera2_nintendo-dsi-traseira']
    for src_name in content:
        objpoints = []
        imgpoints = []
        start = time.time()
        err = calibrate_camera(src_name)
        end = time.time()
        diff = end - start
        print(f"Reprojection error for {src_name} = {err}")
        print(f"Time taken for {src_name} = {diff} seconds")