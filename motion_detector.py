import cv2

'''
Steps:
1. Save the initial frame.
2. Convert this image(without objects) into Gaussian Blur image.
3. Take the frame with objects and convert it into Gaussian Blur.
4. Calculate the difference between both(with and without objects).
5. Define threshold and remove the shadows and other noises.
6. Define borders of the object.
7. Add rectangle around the object.
'''


video = cv2.VideoCapture(0) # it will trigger internal camera, 1 for external camera
first_frame = None

while True:
    check, frame = video.read()  # check: bool if Python can read VideoCapture, frame: numpy array representing first captured image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert each frame into gray scale
    gaussian = cv2.GaussianBlur(gray,(21,21),0) # gray to gaussian blur

    if first_frame is None:
        first_frame = gaussian  # store first frame as soon as camera starts
        continue

    delta = cv2.absdiff(first_frame, gaussian) # diff b/w first and other frames
    threshold = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1] # convert difference value < 30 to black
    threshold = cv2.dilate(threshold, None, iterations=0) # otherwise white
    (cnts,_) = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # define contour area i.e. borders

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('gaussian', gaussian)
    cv2.imshow('delta', delta)
    cv2.imshow('threshold', threshold)
    key = cv2.waitKey(1)  # generates new frame after 1 millisecond
    if key == ord('q'):
        break  # if 'q' is pressed, the will be closed

video.release()

