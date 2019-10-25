import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		snapfil=cv2.imread("image.jpeg")
		snapfilw=int(w)+1
		snapfilh=int(0.37*h)
		snapfil=cv2.resize(snapfil,(snapfilw,snapfilh))
		snapfil_gray=cv2.cvtColor(snapfil,cv2.COLOR_BGR2GRAY)
		_,snap_mask=cv2.threshold(snapfil_gray,25,255,cv2.THRESH_BINARY_INV)
		#cv2.rectangle(img,(x,y),(x+w,int(y+(0.56*h)//2)),(255,0,0),2)
		snap_area=img[y:y+snapfilh , x:x+snapfilw]
		snap_area_no_snap=cv2.bitwise_and(snap_area,snap_area, mask=snap_mask)
		final_snap=cv2.add(snap_area_no_snap,snapfil)

		img[y:y+snapfilh , x:x+snapfilw]=final_snap
		#cv2.imshow("snapfil",snap_area_no_snap)

	cv2.imshow('img',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()