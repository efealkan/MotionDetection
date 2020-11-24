import imutils
import time
import cv2 as cv

video_file = "input/test1.mp4"
video = cv.VideoCapture(0)
time.sleep(2.0)

min_area = 5000
first_frame = None  # Is our background model

while True:
    ret, frame = video.read()

    if frame is None or not ret:
        break

    frame = imutils.resize(frame, width=500)  # Resize the frame
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Get grayscale of the frame
    grayscale = cv.GaussianBlur(grayscale, (21, 21), 0)  # Blur the grayscale image

    if first_frame is None:
        first_frame = grayscale.copy().astype("float")
        continue

    cv.accumulateWeighted(grayscale, first_frame, 0.05)

    difference = cv.absdiff(cv.convertScaleAbs(first_frame), grayscale)  # Larger means there is motion!
    thresh = cv.threshold(difference, 25, 255, cv.THRESH_BINARY)[1]

    thresh = cv.dilate(thresh, None, iterations=2)
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    pos = []

    for c in contours:
        if cv.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv.boundingRect(c)

        centre_x = int(x + (w/2))
        centre_y = int(y + (h/2))

        cv.circle(frame, (centre_x, centre_y), 4, (0, 0, 255), -1)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow("Frame", frame)
    cv.imshow("Whitened", thresh)
    cv.imshow("Frame - Background", difference)
    cv.imshow("Grayscale", grayscale)
    # cv.waitKey(30)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
