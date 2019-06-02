import cv2
import numpy as np 
import imutils
cap = cv2.VideoCapture(0)

while True:
	ret,frame = cap.read()
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(frame, (300, 300))
	print(image.shape)
	cv2.imshow('frame',image)
	# cv2.imshow('gray',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()