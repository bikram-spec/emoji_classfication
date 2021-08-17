# default imports
import tkinter as tk
from tkinter import *
import cv2
import PIL
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import threading


# defining the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
from tensorflow import keras
# new_model=keras.models.load_model('model1.model')
model.load_weights("model.h5")
model.summary()


# defaut variable declaration
# cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "   Angry   ", 1: "Disgust", 2: "  Fear  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
cur_path=os.path.dirname(os.path.abspath(__file__))
emoji_dist={0:cur_path+"/emojis/Angry.png",1:cur_path+"/emojis/Disgust.png",2:cur_path+"/emojis/Fear.png",3:cur_path+"/emojis/Happy.png",4:cur_path+"/emojis/Natural.png",5:cur_path+"/emojis/Sad.png",6:cur_path+"/emojis/Surprised.png"}
# emoji_dist=[]
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap
show_text=[0]
global frame_number

# show_vid function declaration
def show_vid():
    cap=cv2.VideoCapture(0)
    while cap.isOpened():
        executed=False
        ret,frame=cap.read()
        # rezing the image to the (600,500)
        frame = cv2.resize(frame,(600,500),interpolation = cv2.INTER_AREA)

        # cropping and bounding box detection
        # bounding_box = cv2.CascadeClassifier('C:/Users/hp/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        bounding_box=cv2.CascadeClassifier('/home/vikram/Documents/projects/python/cnnprojects/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')

        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            executed=True
            maxindex = int(np.argmax(prediction))
            # show_avatar()
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_text[0]=maxindex

        
        if ret is None:
            print ("Major error!")
        elif ret:
            if(executed):
                global last_frame
                last_frame = frame.copy()
                # print(emoji_dist[show_text[0]])
                print(show_text[0])
                frame2=cv2.imread(emoji_dist[show_text[0]])
                pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)     
                pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
                img = Image.fromarray(pic)
                img2=Image.fromarray(frame2)
                # print(type(img))
                imgtk = ImageTk.PhotoImage(image=img)
                imgtk2=ImageTk.PhotoImage(image=img2)
                lmain.imgtk = imgtk
                lmain2.imgtk2=imgtk2
                lmain.configure(image=imgtk)
                lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
                lmain2.configure(image=imgtk2)
                root.update()
                lmain.after(5, show_vid())
            else:
                pass
        
        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()



if __name__ == '__main__':
    frame_number=0
    root=tk.Tk()
    
    heading=Label(root,text="Photo to Emoji",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')                                 
    heading.pack() 
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain2 = tk.Label(master=root,bd=10)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=670,y=250)
    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
    lmain3.pack()
    lmain3.place(x=900,y=200)
    
    
    root.title("Photo To Emoji")            
    root.geometry("1400x900+100+10") 
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
    show_vid()
    # show_avatar()
    root.mainloop()    