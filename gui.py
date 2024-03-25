import tkinter as tk
from tkinter import *
from tkinter import messagebox 
import tkinter.ttk as ttk
from PIL import ImageTk, Image
import datetime
from glob import glob
import tkinter
import os
import csv
import cv2 
import pandas as pd
import time
import threading

data_dir_name = './dataset/'
def student_frame():
    root1= Toplevel(window)
    root1.geometry("750x250")
    root1.title("Student Entry")
    #Create a Label in New window
    width = 500
    height = 400
    screen_width = root1.winfo_screenwidth()
    screen_height = root1.winfo_screenheight()
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    T = Text(root1, height = 1, width =20)
    l = Label(root1, text = "USN number")
    l.pack()
    T.pack()
    r2 = tk.Button(
    root1,
    text="Capture",
    command=lambda:capture(T),
    bd=10,
    font=("times new roman", 16),
    bg="black",
    fg="yellow",
    height=2,
    width=17,
)
    r2.pack()



   
def capture(text_box,):
    x=text_box.get("1.0","end-1c")
    if len(x.strip())==0:
         messagebox.showinfo("Warning", "Please Enter USN")
    else:
         if os.path.exists(data_dir_name + x):
             messagebox.showinfo("Warning", "USN already exists") 
         else:
             os.mkdir(data_dir_name + x) 
             open_camera(data_dir_name,x)

def open_camera(dir_name,name): 

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("press space to take a photo", 500, 300)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("press space to take a photo", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = dir_name + name +"/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()
    


     

window = Tk()
window.title("Face recognizer")
window.geometry("1280x720")
dialog_title = "QUIT"
dialog_text = "Are you sure want to close?"
window.configure(background="black")

def view_attendance():
    new= Toplevel(window)
    new.geometry("750x250")
    new.title("Attendance")
    #Create a Label in New window
    width = 500
    height = 400
    screen_width = new.winfo_screenwidth()
    screen_height = new.winfo_screenheight()
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    new.geometry("%dx%d+%d+%d" % (width, height, x, y))
    new.resizable(0, 0)
    tree = ttk.Treeview(new, show="headings")
    with open('attendance.csv', 'r', newline='') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)  # Read the header row
                tree.delete(*tree.get_children())  # Clear the current data

                tree["columns"] = header
                for col in header:
                    tree.heading(col, text=col)
                    tree.column(col, width=100)

                for row in csv_reader:
                    tree.insert("", "end", values=row)
    tree.pack(padx=20, pady=20, fill="both", expand=True)
    status_label = tk.Label(new, text="Attendance", padx=20, pady=10)
    status_label.pack()


   


a = tk.Label(
    window,
    text="Face Recognition",
    bg="black",
    fg="yellow",
    bd=10,
    font=("arial", 35),
)
a.pack()


r2 = tk.Button(
    window,
    text="Capture Student",
    command=student_frame,
    bd=10,
    font=("times new roman", 16),
    bg="black",
    fg="yellow",
    height=2,
    width=17,
)
r2.place(x=600, y=520)

r1 = tk.Button(
    window,
    text="View Attendance",
    command=view_attendance,
    bd=10,
    font=("times new roman", 16),
    bg="black",
    fg="yellow",
    height=2,
    width=17,
)
r1.place(x=1000, y=520)

r3 = tk.Button(
    window,
    text="EXIT",
    bd=10,
    command=quit,
    font=("times new roman", 16),
    bg="black",
    fg="yellow",
    height=2,
    width=17,
)
r3.place(x=600, y=660)

window.mainloop()


