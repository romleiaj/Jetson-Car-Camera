import cv2

onboard_cam = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM),"\
" width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)"\
"30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert"\
" ! video/x-raw, format=(string)BGR ! appsink")

while(True):
    ret_val, frame = onboard_cam.read()
    cv2.imshow("myImage", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
