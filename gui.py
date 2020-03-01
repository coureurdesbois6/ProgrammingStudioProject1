from tkinter import filedialog
from PIL import *
from PIL import Image
import numpy as np
import PIL
import tkinter as tk
from PIL import ImageTk
import ImageReader as ir

method = ir.HU

def searchdirectory(img, imgbox, dirtextbox, ermsgbox=None, size=(300, 400)):
    directory = filedialog.askopenfilename()
    dirtextbox.delete(0, len(dirtextbox.get()))
    dirtextbox.insert(0, directory)
    loaddirectory(img, dirtextbox.get(), imgbox, ermsgbox=ermsgbox, size=size)

def reset():
    size = 0

    if method == ir.HU:
        size = 7
    elif method == ir.R:
        size = 10
    elif method == ir.ZERNIKE:
        size = 12
    momentsdb = np.zeros(shape=(1, 36, size), dtype=float)

    np.save(method, momentsdb)


def loaddirectory(imge, dir, imgbox, ermsgbox=None, size=(300, 400)):
    try:
        pilimg = Image.open(dir)
        pilimg.thumbnail(size, PIL.Image.ANTIALIAS)
        imge = ImageTk.PhotoImage(pilimg)
        imgbox.config(image=imge)
        imgbox.image = imge
        if isinstance(ermsgbox, tk.Label):
            ermsgbox.config(text="Successfully opened file", fg="green")
    except:
        if isinstance(ermsgbox, tk.Label):
            ermsgbox.config(text="CANNOT OPEN FILE", fg="red")


def launch():
    global method
    global dirtextbox
    try:
        im = Image.open(dirtextbox.get())
        print("working...")
        ir.ImageReader().launch(dirtextbox.get(), methoddb=method)
    except:        # TODO: make exception specific
        direrrmsg.config(text="CANNOT OPERATE ON FILE", fg="red")


def savesample(chardirtextbox, character):
    image = Image.open(chardirtextbox.get())
    imagereader = ir.ImageReader()
    image = imagereader.to_greyscale(image)
    rectangles = imagereader.rectangles(imagereader.blob_coloring_8_connected(np.asarray(image)))
    charimg = image.crop(
        (rectangles[0][1] - 1, rectangles[0][0] - 1, rectangles[0][3] + 2, rectangles[0][2] + 2)).resize(
        ir.SAMPLE_SHAPE)
    #charimg.show()  # show greyscale image, unnecessary.
    imagereader.storesample(np.asarray(charimg), character, methoddb=method)
    print("Saved", character)


def onselect(evt, textbox):
    w = evt.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    number = ir.ImageReader().getsamplecount(value)
    text = value + " has " + str(number) + " samples stored."
    textbox.config(text=text)


def savesampleset(chardirtextbox, orderbox):
    order = orderbox.get()
    image = Image.open(chardirtextbox.get())
    imagereader = ir.ImageReader()
    image = imagereader.to_greyscale(image)
    rectangles = imagereader.rectangles(imagereader.blob_coloring_8_connected(np.asarray(image)))
    print(len(rectangles))
    for i in range(len(rectangles)):
        charimg = image.crop(
            (rectangles[i][1] - 1, rectangles[i][0] - 1, rectangles[i][3] + 2, rectangles[i][2] + 2)).resize(
            ir.SAMPLE_SHAPE)
        imagereader.storesample(np.asarray(charimg), order[i], methoddb=method)
    print("Saved set")


def options_window():  # use .state to check if it is already up
    global charimg
    options = tk.Toplevel(root)

    scrollframe = tk.Frame(options)
    infoframe = tk.Frame(options, padx=20)

    scrollbar = tk.Scrollbar(scrollframe)
    characterslist = tk.Listbox(scrollframe, yscrollcommand=scrollbar.set)
    for i in range(10):
        characterslist.insert(tk.END, str(int(i)))
    for i in range(26):
        characterslist.insert(tk.END, chr(65 + int(i)))

    scrollbar.config(command=characterslist.yview)

    characterslist.pack(side=tk.LEFT, fill=tk.BOTH)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    infotext = tk.Label(infoframe)
    characterslist.bind('<<ListboxSelect>>', lambda e: onselect(e, infotext))
    infotext.pack()

    middleframe = tk.Frame(options)

    searchframe = tk.Frame(middleframe)

    charimgframe = tk.Frame(options, height=50, width=50, padx=40)

    pilimg1 = Image.open("sample.jpg")
    pilimg1.thumbnail(ir.SAMPLE_SHAPE, PIL.Image.ANTIALIAS)
    charimg = ImageTk.PhotoImage(pilimg1)
    charimagebox = tk.Label(charimgframe, image=charimg)
    charimagebox.image = charimg  # keep img as reference to avoid a certain bug
    charimagebox.pack(expand=1)

    chardirtextbox = tk.Entry(searchframe, text="")
    chardirselect = tk.Button(searchframe, text="Search",
                              command=lambda: searchdirectory("charimg", charimagebox, chardirtextbox,
                                                              ermsgbox=chardirerrmsg, size=ir.SAMPLE_SHAPE))
    chardirload = tk.Button(searchframe, text="Load",
                            command=lambda: loaddirectory("charimg", dirtextbox, chardirtextbox.get(),
                                                          ermsgbox=chardirerrmsg, size=ir.SAMPLE_SHAPE))
    chardirerrmsg = tk.Label(searchframe, text="")

    chardirtextbox.grid(row=0, column=0)
    chardirselect.grid(row=0, column=1)
    chardirload.grid(row=0, column=2)
    chardirerrmsg.grid(row=1, column=0)

    searchframe.pack()
    methodstropt = tk.StringVar(launchframe)
    methodstropt.set("HU")
    methodselectopt = tk.OptionMenu(middleframe, methodstropt, "HU", "R", "ZERNIKE", command=method_select)
    savebutton1 = tk.Button(middleframe, text="Save sample", command=lambda: savesample(chardirtextbox, characterslist.get(characterslist.curselection())))
    orderbox = tk.Entry(middleframe)
    orderbox.insert(tk.END, "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")
    infotext2 = tk.Label(middleframe, text="Enter the order of characters below")
    savebutton2 = tk.Button(middleframe, text="Save sample set", command=lambda: savesampleset(chardirtextbox, orderbox))
    methodselectopt.pack()
    savebutton1.pack()
    savebutton2.pack()
    infotext2.pack()
    orderbox.pack()
    resetbutton = tk.Button(options, text="Reset samples", command=lambda: reset())

    scrollframe.grid(row=0, column=0)
    infoframe.grid(row=0, column=1)
    charimgframe.grid(row=0, column=2)
    resetbutton.grid(row=1, column=2)
    middleframe.grid(row=0, column=3)


def method_select(value):
    global method
    if value == "HU":
        method = ir.HU
    elif value == "R":
        method == ir.R
    elif value == "ZERNIKE":
        method == ir.ZERNIKE


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
dirtextbox = tk.Entry(dirframe, text="")
dirselect = tk.Button(dirframe, text="Search",
                      command=lambda: searchdirectory("img", imagebox, dirtextbox, ermsgbox=direrrmsg))
dirload = tk.Button(dirframe, text="Load",
                    command=lambda: loaddirectory("img", dirtextbox.get(), imagebox, ermsgbox=direrrmsg))
direrrmsg = tk.Label(dirframe, text="")

dirtextbox.grid(row=0, column=0)
dirselect.grid(row=0, column=1)
dirload.grid(row=0, column=2)
direrrmsg.grid(row=1, column=0)

dirframe.grid(row=0, columnspan=2)

methodstr = tk.StringVar(launchframe)
methodstr.set("HU")

methodselect = tk.OptionMenu(launchframe, methodstr, "HU", "R", "ZERNIKE", command=method_select)
launchbutton = tk.Button(launchframe, text="Launch", command=lambda: launch())

methodselect.grid(row=1, column=0)
launchbutton.grid(row=1, column=1)

launchframe.pack()

bottomframe = tk.Frame(userframe)

optionsframe = tk.Frame(bottomframe, bd=20)
optionsbutton = tk.Button(optionsframe, text="Edit lookup database", command=lambda: options_window())
optionsbutton.pack()
optionsframe.grid(row=0, column=0)
bottomframe.pack()

#                   #
#   IMAGE FRAME     #
#                   #
imgframe = tk.Frame(root, height=400, width=300)
pilimg = Image.open("sample.jpg")
pilimg.thumbnail((300, 400), PIL.Image.ANTIALIAS)
img = ImageTk.PhotoImage(pilimg)
imagebox = tk.Label(imgframe, image=img)
imagebox.image = img  # keep img as reference to avoid a certain bug


imagebox.pack(expand=1)
imgframe.pack_propagate(False)

imgframe.grid(row=0, column=0)
userframe.grid(row=0, column=1)
# imagebox.bind('<Return>', func)
root.mainloop()
