import cv2

def show_webcam(mirror=False, width=600, height=600):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        cv2.namedWindow('my webcam',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('my webcam', width, height)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    #cv2.destroyAllWindows()

show_webcam()
