import cv2
import threading
import numpy as np


class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)


    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def run_cameras():
    window_title = "Test_six_Cameras"
    first_camera = CSI_Camera()
    first_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            capture_width=1920,
            capture_height=1080,
            flip_method=2,
            display_width=960,
            display_height=540,
        )
    )
    first_camera.start()

    second_camera = CSI_Camera()
    second_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            capture_width=1920,
            capture_height=1080,
            flip_method=2,
            display_width=960,
            display_height=540,
        )
    )
    second_camera.start()
    
    third_camera = CSI_Camera()
    third_camera.open(
        gstreamer_pipeline(
            sensor_id=2,
            capture_width=1920,
            capture_height=1080,
            flip_method=2,
            display_width=960,
            display_height=540,
        )
    )
    third_camera.start()
    
    fourth_camera = CSI_Camera()
    fourth_camera.open(
        gstreamer_pipeline(
            sensor_id=3,
            capture_width=1920,
            capture_height=1080,
            flip_method=2,
            display_width=960,
            display_height=540,
        )
    )
    fourth_camera.start()
    
    fifth_camera = CSI_Camera()
    fifth_camera.open(
        gstreamer_pipeline(
            sensor_id=4,
            capture_width=1920,
            capture_height=1080,
            flip_method=2,
            display_width=960,
            display_height=540,
        )
    )
    fifth_camera.start()
    
    sixth_camera = CSI_Camera()
    sixth_camera.open(
        gstreamer_pipeline(
            sensor_id=5,
            capture_width=1920,
            capture_height=1080,
            flip_method=2,
            display_width=960,
            display_height=540,
        )
    )
    sixth_camera.start()

    if first_camera.video_capture.isOpened() and second_camera.video_capture.isOpened() and third_camera.video_capture.isOpened() and fourth_camera.video_capture.isOpened() and fifth_camera.video_capture.isOpened() and sixth_camera.video_capture.isOpened():

        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                _, first_image = first_camera.read()
                _, second_image = second_camera.read()
                _, third_image = third_camera.read()
                _, fourth_image = fourth_camera.read()
                _, fifth_image = fifth_camera.read()
                _, sixth_image = sixth_camera.read()
                # Use numpy to place images next to each other
                camera_images_up = np.hstack((first_image, second_image, third_image)) 
                camera_images_down = np.hstack((fourth_image, fifth_image, sixth_image)) 
                camera_images_full = np.vstack((camera_images_up, camera_images_down))
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, camera_images_full)
                else:
                    break

                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
        finally:

            first_camera.stop()
            first_camera.release()
            second_camera.stop()
            second_camera.release()
            third_camera.stop()
            third_camera.release()
            fourth_camera.stop()
            fourth_camera.release()
            fifth_camera.stop()
            fifth_camera.release()
            sixth_camera.stop()
            sixth_camera.release()
            
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to open both cameras")
        first_camera.stop()
        first_camera.release()
        second_camera.stop()
        second_camera.release()
        third_camera.stop()
        third_camera.release()
        fourth_camera.stop()
        fourth_camera.release()
        fifth_camera.stop()
        fifth_camera.release()
        sixth_camera.stop()
        sixth_camera.release()



if __name__ == "__main__":
    run_cameras()
