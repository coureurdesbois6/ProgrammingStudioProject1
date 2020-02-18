from tkinter import filedialog
from PIL import *
from PIL import Image
import numpy as np
import PIL
import tkinter as tk
from PIL import ImageTk
import ImageReader as ir



def searchdirectory():
    global dirtextbox
    directory = filedialog.askopenfilename()
    dirtextbox.delete(0,len(dirtextbox.get()))
    dirtextbox.insert(0,directory)
    loaddirectory()

def loaddirectory():
    global dirtextbox
    global direrrmsg
    global img
    try:
        pilimg = Image.open(dirtextbox.get())
        pilimg.thumbnail((300, 400), PIL.Image.ANTIALIAS)
        img = ImageTk.PhotoImage(pilimg)
        imagebox.config(image=img)
        direrrmsg.config(text="Successfully opened file", fg="green")
    except:
        direrrmsg.config(text="CANNOT OPEN FILE", fg="red")

def launch():
    global dirtextbox
    try:
        im = Image.open(dirtextbox.get())
        print("working...")
        ir.ImageReader(dirtextbox.get()).launch()
    except:
        direrrmsg.config(text="CANNOT OPERATE ON FILE", fg="red")



root = tk.Tk()

#                   #
#   USER FRAME      #
#                   #
userframe = tk.Frame(root)


#                   #
#   LAUNCH FRAME    #
#                   #
launchframe = tk.Frame(userframe)


#                       #
#   DIRECTORY FRAME     #
#                       #
dirframe = tk.Frame(launchframe)
dirtextbox = tk.Entry(dirframe, text="hello")
dirselect = tk.Button(dirframe, text="Search", command= lambda: searchdirectory())
dirload = tk.Button(dirframe, text="Load", command= lambda: loaddirectory())
direrrmsg = tk.Label(dirframe, text="")

dirtextbox.grid(row=0, column=0)
dirselect.grid(row=0, column=1)
dirload.grid(row=0, column=2)
direrrmsg.grid(row=1, column=0)

dirframe.grid(row=0, columnspan=2)


method = tk.StringVar(launchframe)
method.set("Hu Moments")
methodselect = tk.OptionMenu(launchframe, method, "Hu Moments", "Zernike Moments", "PLACEHOLDER")

launchbutton = tk.Button(launchframe, text="Launch", command= lambda: launch())


methodselect.grid(row=1, column=0)
launchbutton.grid(row=1, column=1)

launchframe.pack()


#                   #
#   IMAGE FRAME     #
#                   #
imgframe = tk.Frame(root, height=400, width=300)
#pilimg = Image.open("sample.jpg")
pilimg = Image.open("sample.jpg")
pilimg.thumbnail((300,400), PIL.Image.ANTIALIAS)
img = ImageTk.PhotoImage(pilimg)
imagebox = tk.Label(imgframe, image=img)
imagebox.image = img #keep img as reference to avoid a certain bug

imagebox.pack(expand=1)
imgframe.pack_propagate(False)







imgframe.grid(row=0, column=0)
userframe.grid(row=0, column=1)
#imagebox.bind('<Return>', func)
root.mainloop()