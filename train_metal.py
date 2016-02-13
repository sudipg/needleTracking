"""
GUI to go through different images to store a comprehensive list of RGB
values for metal surfaces
"""


from Tkinter import *
import tkFileDialog
import numpy as np
from scipy import misc, ndimage
import cv2
from matplotlib import pyplot as plt
import sys
import math
import matplotlib.image as mpimg
from PIL import Image, ImageTk
import pdb
import copy

class ColorSampler():
	"""
	class of tkinter App for collecting samples from userselected images
	to train for image segmentation
	"""

	def __init__(self, master):
		"""
		Initialize and setup the GUI
		"""
		self.root = master

		self.filename = '' # the file to perform all operations on

		menubar = Menu(self.root)
		menubar.add_command(label="open", command=self.open_file)
		menubar.add_command(label="save", command=self.save_file)
		master.config(menu=menubar)

		# frame to hold all the UI components
		frame = Frame(self.root)
		frame.grid(row=0,column=0) # fill all

		self.imgToSample = None # this is the scipy img we work with

		self.imgDisplayed = None
		self.imgHeight = 0
		self.imgWidth = 0

		# the following few lines initialize the canvas where the image is drawn
		self.cframe = Frame(frame)
		self.canvas = Canvas(self.cframe,width=1200,height=800)		
		self.canvas.grid(row=0,column=0)
		self.cframe.grid(row=0,column=0)
		self.hbar=Scrollbar(self.cframe,orient=HORIZONTAL)
		self.hbar.grid(row=1,column=0, sticky=W+E)
		self.vbar=Scrollbar(self.cframe,orient=VERTICAL)
		self.vbar.grid(row=0,column=1, sticky=N+S)
		self.vbar.config(command=self.canvas.yview)
		self.hbar.config(command=self.canvas.xview)
		self.canvas.config(width=1200,height=800)
		self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
		self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

		# We bind the canvas to respond to event of mouse button click 
		self.canvas.bind("<Button 1>", self.get_sample) 

		# We initialize a list [n,3] to store the rgb values

	def open_file(self):
		"""
		open a specified file
		"""
		self.filename = tkFileDialog.askopenfilename(parent=self.root)
		print 'The file image you selected is: '+self.filename
		self.canvas.delete(self.imgDisplayed) # remove any old image
		self.labels = [] # wipe records from previous image

		# setup to display the image
		# NOTE: need to go through this format for Tk
		f = Image.open(self.filename)
		photo = ImageTk.PhotoImage(f)
		self.imgDisplayed = self.canvas.create_image(0,0,image=photo, anchor='nw', state=NORMAL)
		self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set,scrollregion=(0, 0, photo.width(), photo.height()))
		self.canvas.image = photo 
		self.canvas.grid(row=0,column=0)

		self.imgToSample = misc.imread(self.filename)

	def get_sample(self,event):
		canvas = event.widget
		x = canvas.canvasx(event.x)
		y = canvas.canvasy(event.y)
		x,y = int(x),int(y) # need to parse to integers so we can access RGB values at index
		canvas.create_oval(x-1,y-1,x+1,y+1, fill='red') # show sample dot

		#sample
		sample = self.imgToSample[y,x]

		# extract RGB values
		self.labels.append(sample)
		print str(x) + " " + str(y) + " " + str(sample)

	def save_file(self):
		targetFileName = self.filename.split(".")[0]+".data"
		f = open(targetFileName, 'w+')
		for sample in self.labels:
			f.write(str(sample[0])+","+str(sample[1])+","+str(sample[2])+'\n')
		print("data (RGB) saved at "+targetFileName)
		f.close()



def main():
	root = Tk()
	app = ColorSampler(root)
	root.mainloop()



if __name__ == '__main__':
    main()
