import cv2
import face_recognition
import numpy as np
import os
import mysql.connector
from datetime import datetime, date

import pyttsx3

import time
import os
engine=pyttsx3.init()
path='pro_images'
images=[]
classNames=[]
myList=os.listdir(path)
#print(myList)
mysqldb = mysql.connector.Connect(host="localhost", user='root', password="", database="student")
mycursor = mysqldb.cursor()

for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(name):
    now = datetime.now()
    oridate = now.strftime("%Y-%m-%d")
    sql = "SELECT EXISTS (SELECT * from smartattend WHERE name = %s and date=%s);"
    mycursor.execute(sql,(name,oridate))
    res = mycursor.fetchone()
    res1 = res[0]
    if (res1==1):
        saying = "attendence already recorded"
        engine.say(saying)
        engine.runAndWait()
    else:
        sql = "INSERT INTO smartattend(name,date) VALUES (%s, %s)"
        val = (name, oridate)
        mycursor.execute(sql, val)
        mysqldb.commit()
        saying = "attendence taken"
        engine.say(saying)
        engine.runAndWait()

encodeListKnown=findEncodings(images)
print('encoding complete')

cap=cv2.VideoCapture(0)
while True:
    sucess,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encodesCurFrame = face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            #sql = "SELECT EXISTS (SELECT * from smartattend WHERE name = name);"
            #mycursor.execute(sql)
            #res = mycursor.fetchone()
            #res1 = res[0]
            #markAttendence(name,res1)
            markAttendence(name)
        else:
            name = "unknown"
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n [INFO] Exiting Program and cleanup stuff")
        cap.release()
        cv2.destroyAllWindows()
        break


