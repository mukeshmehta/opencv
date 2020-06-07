import cv2

video = cv2.VideoCapture(0) # it will trigger internal camera, 1 for external camera
face_cascade = cv2.CascadeClassifier("C:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")

f = 1
while True:
    f += 1
    check, frame = video.read() # check: bool if Python can read VideoCapture, frame: numpy array representing first captured image
    #print(check)
    print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert each frame into gray scale
    #cv2.imshow('Captured', gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)
    if check:   # if face is present: put it in rectangle
        for x, y, w, h in faces:
            gray = cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("Face", gray)
    key = cv2.waitKey(1) # generates new frame after 1 millisecond
    if key == ord('q'):
        break   # if 'q' is pressed, the will be closed
print(f) # number of frames
video.release()
cv2.destroyAllWindows()