# Code adapted from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html and # code adapted from: https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
import numpy as np
import cv2 as cv
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# This function iterates over all images in the training folder and attempts to find the chessboard corners manually.
# If it can't find them it opens a window to add them manually. It then saves the object and image points.
# Then it performs camera calibration if the remove_bad_images is set to True. It then calculates the re-projection error after the addition of the training image
# and compares it to the re-projection error before adding this image. If the error increases, the object and image points are removed again and the image is not used.
# This function returns the object points and image points of the 'good' images (if remove_bad_features = False, it saves them from all images). 
def find_chessboard_corners(image_folder, remove_bad_images=False):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chessboard square size (mm)
    square_size = 16.6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * square_size

    # Get the list of image files
    images = glob.glob(os.path.join(image_folder, '*.jpeg'))

    # Initial calibration 
    ret, mtx, dst, rvecs, tvecs = None, None, None, None, None
    prev_error = float('inf')

    #iterate over images to find chessboard corners and add image and object points automatically if possible, otherwise manually.
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (9,6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

            # If bad images need to be excluded, calculate reprojection error after addition of each image, if it increases, dont use the image.
            if remove_bad_images:
                # Calibrate camera with the new image
                ret_new, mtx_new, dst_new, rvecs_new, tvecs_new = calibrate_camera(objpoints, imgpoints, gray.shape[::-1])

                # Calculate reprojection error
                error = calculate_reprojection_error(objpoints, imgpoints, rvecs_new, tvecs_new, mtx_new, dst_new)

                # Check if the error increased
                if ret_new and error >= prev_error:
                    # Remove the last added image
                    objpoints.pop()
                    imgpoints.pop()
                else:
                    # Update calibration results
                    ret, mtx, dst, rvecs, tvecs = ret_new, mtx_new, dst_new, rvecs_new, tvecs_new
                    prev_error = error
            else:
                ret, mtx, dst, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, gray.shape[::-1])

        else:
            #if not found, set cornerpoints to mouseclicks, adding them as image points and object points.
            cv.imshow('img', img)
            cv.setMouseCallback('img', click_event)
            cv.waitKey(0)

            if remove_bad_images:
                # Calibrate camera with the new image
                ret_new, mtx_new, dst_new, rvecs_new, tvecs_new = calibrate_camera(objpoints, imgpoints, gray.shape[::-1])

                # Calculate reprojection error
                error = calculate_reprojection_error(objpoints, imgpoints, rvecs_new, tvecs_new, mtx_new, dst_new)

                # Check if the error increased
                if ret_new and error >= prev_error:
                    # Remove the last added image
                    objpoints.pop()
                    imgpoints.pop()
                else:
                    # Update calibration results
                    ret, mtx, dst, rvecs, tvecs = ret_new, mtx_new, dst_new, rvecs_new, tvecs_new
                    prev_error = error
            else:
                ret, mtx, dst, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, gray.shape[::-1])
    cv.destroyAllWindows()
    return objpoints, imgpoints

# Helper function to manually select cornerpoints. It shows the coordinates of the clicked points and saves them as image points.
def click_event(event, x, y, flags, param, img, objp, imgpoints, objpoints):
    if event == cv.EVENT_LBUTTONDOWN:

        # displaying the coordinates on the image
        print(f"Clicked coordinates: ({x}, {y})")
        cv.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv.imshow('img', img)

        # saving the corner points as image points and appending the corresponding object point.
        imgpoints.append((x, y))
        objpoints.append(objp)

# This function calibrates the camera based on the object points and image points and returns the calibration settins (camera matrix, distortion matrix, etc.)
def calibrate_camera(objpoints, imgpoints, shape):
    # getting the camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dst, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, None, None)
    return ret, mtx, dst, rvecs, tvecs

# This function undistorts the images using the calibration setting and saves the undistorted images in a new folder.
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

# This function calculates the mean re-projection error by re-projecting the image points using the object points, rvecs, tvecs, mtx, and dst and comparing it to the earlier image points.
def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dst):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dst)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    return mean_error / len(objpoints)

# This function plots the camera positions of all training images by plotting the translation vectors as points and then adding a quiver based on the rotation vector.
def plot_camera_positions(image_folder, rvecs, tvecs):
    # create 3d figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For each tvec, add a scatter based on x[0], y[1], z[2] coordinates.
    for i, (tvec, rvec) in enumerate(zip(tvecs, rvecs)):
        ax.scatter(tvec[0], tvec[1], tvec[2], c='r', marker='o')  # camera position
        ax.text(tvec[0], tvec[1], tvec[2], image_folder[i]) #train image annotation
        ax.quiver(tvec[0], tvec[1], tvec[2], rvec[0], rvec[1], rvec[2], length=0.1, color='b')  # rotation vector

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera position of training images')
    plt.show()

def main(remove_bad_images = False):
    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the image folder
    image_folder = os.path.join(current_dir, "train images")

    #create directory to store undistorted images
    undistorted_folder = os.path.join(current_dir, "undistorted_images")
    os.makedirs(undistorted_folder, exist_ok=True)

    #get obj and img points
    objpoints, imgpoints, gray, ret, mtx, dst, rvecs, tvecs = find_chessboard_corners(image_folder, remove_bad_images)

    #calibrate camera and save camera matrix and dst coefficients
    ret, mtx, dst, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, gray.shape[::-1])
    np.savetxt('camera_matrix.txt', mtx)
    np.savetxt('distortion_coefficients.txt', dst)

    #undistort images
    undistort_images(image_folder, undistorted_folder, mtx, dst)

    #calculate re-projection error
    mean_error = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dst)
    print("total error:", mean_error)

    # plot camera positions
    plot_camera_positions(image_folder, rvecs, tvecs)

    
if __name__ == "__main__":
    main(remove_bad_images=False)