import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('path to .mp4')
success,image = vidcap.read()
count = 0
while success:
  img = image
  #resccale img
  scale_percent = 15.35 # percent of original size
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)
  
  #resize image
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
  #crop the scaled img
  cropped_image = resized[0:400, 129:460]
  print('Resized Dimensions : ',cropped_image.shape)

  cv2.imwrite("path_to_imagefoler/frame%d.png" % count, cropped_image)     #save frame as JPEG file
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  count += 1
