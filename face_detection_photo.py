import cv2

img = cv2.imread("kohli.jpg",1)
#print(img.shape)
img2 = cv2.imread("kohli.jpg",0)
#resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

face_cascade = cv2.CascadeClassifier("C:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.15, minNeighbors=5)
print(type(faces))
print(faces)

for x,y,w,h in faces:
    face_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

cv2.imshow("Captain",face_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
