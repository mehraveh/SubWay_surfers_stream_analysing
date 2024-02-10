import cv2
vidcap = cv2.VideoCapture('/Users/mehravehahmadi/yolov5/data/images/newV.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("newversion/new%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1