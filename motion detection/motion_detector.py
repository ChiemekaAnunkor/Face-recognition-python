import cv2, time
from datetime import datetime
import pandas

# creates a vidoe object to record using webcam, defult is 0
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

first_frame = None
status_list = [None, None]
times=[]

#creates our pandas data clumns
df=pandas.DataFrame(columns=["Start", "End"])

while True:
    check, frame = video.read()
    status=0

    #coverts vidoe to gray scale and gussians 
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

# this is used to store the first image 
    if first_frame is None:
        first_frame=gray
        continue

    #calculates the difference between the first frame and others 
    delta_frame=cv2.absdiff(first_frame,gray)

    #creates a threshold of 30, if greater than 30 then is it white and has a square, and if less then then is black
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    #defines the border/boundaries of the items in frame threshold.
    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # removes noise and shadows, keeps frames with greater than 1000
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status=1

        #create the dimensions of the green rectangle
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

    # creates list of status for every frame 
    status_list.append(status)
    status_list=status_list[-2:]

    #record datetime in a list when a change in the frame occurs. i.e motion

    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    if status_list[-1]==1 and status_list[-2]==0:
            times.append(datetime.now())

# displays the fames on screen
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Thresh Frame", thresh_frame)
    cv2.imshow("Color Frame",frame)

    # defines frame rate, so new frame ever 1 mili sec
    key = cv2.waitKey(1)

    #press button to close and add values to csv file.
    if key==ord('x'):
        if status==1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

#stores time values in a dataphrame
for i in range(0, len(times), 2):
    df=df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")

# close our windows
video.release()
cv2.destroyAllWindows()
