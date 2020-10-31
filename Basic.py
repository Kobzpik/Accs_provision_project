import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('ImageBasic/elon-musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgBill = face_recognition.load_image_file('ImageBasic/bill-gates.jpg')
imgBill = cv2.cvtColor(imgBill,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocBill = face_recognition.face_locations(imgBill)[0]
encodeBill=face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill,(faceLocBill[3],faceLocBill[0],faceLocBill[1],faceLocBill[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeBill],encodeElon)
print(result)


cv2.imshow('elon musk',imgElon)
cv2.imshow('bill gates',imgBill)
cv2.waitKey(0)