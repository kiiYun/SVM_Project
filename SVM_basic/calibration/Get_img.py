import cv2
import numpy as py
import os


def gstreamer_pipeline(
    sensor_id=5,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
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


def show_camera():
    window_title = "Get_Image"

    num = 0
    
    print(gstreamer_pipeline())
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
  
            while True:
                ret_val, frame = video_capture.read()
                
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:

                    cv2.imshow(window_title,frame)

                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF

                if keyCode == 27 or keyCode == ord('q'):
                    break
                if keyCode == ord('s'):
                    cv2.imwrite('./calib_img_5/img' + str(num) + '.png', frame)
                    print("image saved!")
                    num += 1

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == '__main__':
    show_camera()
