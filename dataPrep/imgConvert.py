import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('videoFolder/penVid.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("imgList/frame%d.png" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  count += 1
