# System dependencies
import time
from datetime import datetime
import random
import numpy as np

import ctypes
import win32gui, win32ui, win32con, win32api, win32process, win32com.client

import threading
from multiprocessing import Process, Manager, Queue

from tkinter import *

import ctypes
import ctypes.wintypes

import cv2

# from pynput import keyboard
import keyboard

# Custom dependencies
from Tool.param_menu import ParamMenu
from Environment.read_screen import ReadScreen
