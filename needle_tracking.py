"""
Needle tracking
"""
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt 
import cv2
import pdb
from Tkinter import *
import tkFileDialog
import sys

use_old_file = int(sys.argv[1])

filename = 'images/left1004.jpeg'


if not use_old_file:
	root = Tk()
	filename = tkFileDialog.askopenfilename(parent = root)
	root.destroy()

img = misc.imread(filename)
misc.imshow(img)
