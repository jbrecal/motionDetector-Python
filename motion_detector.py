import cv2, time, pandas
from datetime import datetime

first_frame = None #stores the first frame
status_list = [None,None] #stores the status of the motion
times = []
df=pandas.DataFrame(columns=["Start", "End"]) #stores the start and end times of the motion


video=cv2.VideoCapture(0) #stores the video

while True: 
    check, frame=video.read() #stores the frame
    status = 0 #stores the status of the motion
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts the frame to grayscale
    gray=cv2.GaussianBlur(gray,(21,21),0) #blurs the grayscale frame

    if first_frame is None: #checks if the first frame is None
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame, gray) #calculates the difference between the first frame and the current frame
    threshold_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1] #converts the difference frame to binary
    threshold_frame=cv2.dilate(threshold_frame, None, iterations=2) #dilates the binary frame
    
    (cnts,_) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finds the contours

    for contour in cnts:
        if cv2.contourArea(contour) < 10000: 
            continue
        status=1 #sets the status to 1
        (x,y,w,h)=cv2.boundingRect(contour) #finds the bounding rectangle
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3) #draws the bounding rectangle
        
    status_list.append(status) #appends the status to the status list
    if status_list[-1]==1 and status_list[-2]==0: #checks if the status is 1 and the previous status is 0
        times.append(datetime.now()) #appends the time to the times list
        print("Motion detected at: ", times)
    if status_list[-1]==0 and status_list[-2]==1: #checks if the status is 0 and the previous status is 1
        times.append(datetime.now()) #appends the time to the times list
        print("Motion ended at: ", times)   
               
    #cv2.imshow("Gray Frame", gray) #shows the grayscale frame
    #cv2.imshow("Delta Frame", delta_frame) #shows the difference frame
    #cv2.imshow("Threshold Frame", threshold_frame) #shows the threshold frame
    cv2.imshow("Color Frame", frame) #shows the color frame

    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    
    
print(status_list)   
print(times) 

rows = [] #stores the start and end times of the motion

for i in range(0, len(times), 2): 
    rows.append({"Start": times[i], "End": times[i + 1]})  #appends the start and end times of the motion

# Use pd.concat to append the new rows to the DataFrame
df = pandas.concat([df, pandas.DataFrame(rows)], ignore_index=True) 

df.to_csv("Times.csv")  # saves the dataframe to a csv file  

video.release()
cv2.destroyAllWindows()
    