# Code adapted from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html and # code adapted from: https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
import numpy as np
import cv2 as cv
import glob
import os

def find_chessboard_corners(image_folder):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chessboard square size (mm)
    square_size = 16.6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) * square_size

    # Get the list of image files
    images = glob.glob(os.path.join(image_folder, '*.jpeg'))

    #iterate over images to find chessboard corners and add image and object points automatically if possible, otherwise manually.
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
        else:
            #if not found, set cornerpoints to mouseclicks, adding them as image points and object points.
            cv.imshow('img', img)
            cv.setMouseCallback('img', click_event)
            cv.waitKey(0)
    cv.destroyAllWindows()
    return objpoints, imgpoints

def click_event(event, x, y, flags, param, img, objp, imgpoints, objpoints):
    if event == cv.EVENT_LBUTTONDOWN:

        # displaying the coordinates on the image
        print(f"Clicked coordinates: ({x}, {y})")
        cv.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv.imshow('img', img)

        # saving the corner points as image points and appending the corresponding object point.
        imgpoints.append((x, y))
        objpoints.append(objp)

def calibrate_camera(objpoints, imgpoints, shape):
    # getting the camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, None, None)
    return ret, mtx, dist, rvecs, tvecs

def undistort_images(image_folder, undistorted_folder, mtx, dst):
    images = glob.glob(os.path.join(image_folder, '*.jpeg'))
    for fname in images:
        img = cv.imread(fname)
        h, w = img.shape[:2]

        # performing optimal camera matrix calculation
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))

        # undistorting
        dst = cv.undistort(img, mtx, dst, None, newcameramtx)

        # cropping
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # saving the images
        output_path = os.path.join(undistorted_folder, 'undistorted_' + os.path.basename(fname))
        cv.imwrite(output_path, dst)

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    return mean_error / len(objpoints)

def main():
    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the image folder
    image_folder = os.path.join(current_dir, "train images")

    #create directory to store undistorted images
    undistorted_folder = os.path.join(current_dir, "undistorted_images")
    os.makedirs(undistorted_folder, exist_ok=True)

    #get obj and img points
    objpoints, imgpoints, gray = find_chessboard_corners(image_folder)

    #calibrate camera and save camera matrix and dst coefficients
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, gray.shape[::-1])
    np.savetxt('camera_matrix.txt', mtx)
    np.savetxt('distortion_coefficients.txt', dist)

    #undistort images
    undistort_images(image_folder, undistorted_folder, mtx, dist)

    #calculate re-projection error
    mean_error = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print("total error:", mean_error)

    
if __name__ == "__main__":
    main()