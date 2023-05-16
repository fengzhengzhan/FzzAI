# -*- coding: utf-8 -*-
# print("[*] import dependencies")

######################
# System dependencies
######################
import abc

import time
from datetime import datetime

import win32gui, win32ui, win32con, win32api, win32process, win32com.client
from pynput import keyboard, mouse

import os
import random
import pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter

import threading
from multiprocessing import Process, Manager, Queue

import sqlite3
from logging import *

import ctypes
import ctypes.wintypes

# from reloading import reloading

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

######################
# Custom dependencies
######################
# 依赖只能跨三个文件，无法跨越4个文件，因此将自定义模块从依赖导入包中删除，防止循环引用
import Tool.project_path as projectpath
from Tool.logger import projlog

import conf as confglobal
import Module.game_action_HollowKnight_fzz.config_hollowknight as confhk
