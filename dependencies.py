# -*- coding: utf-8 -*-

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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

######################
# Custom dependencies
######################

from Agent.action import Action
from Agent.listener import ProcessListenerKeyboard
from Agent.agent import Agent

from Environment.transport_manager import ProcessTransportManager
from Environment.read_screen import ProcessReadScreen
from Environment.environment import Environment
from Environment.change_env import ChangeEnv

import Module.game_action_HollowKnight_fzz.config_hollowknight as confhk
from Module.game_action_HollowKnight_fzz.boss_hornet.network_CNN import NetworkCNN
from Module.game_action_HollowKnight_fzz.keybindings_hollowknight import PrepareBindings, OperationBindings
from Module.game_action_HollowKnight_fzz.traintest import ModelTrainTest

import Tool.project_path as projectpath
from Tool.status_window import GlobalStatus, StatusWindow
from Tool.param_menu import ParamMenu
from Tool.logger import projlog

import conf as confglobal
