# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:32:50 2021

@author: landehuxi
"""
from PIL import Image
import numpy as np
import numpy as np  # numpy库
import xlrd
import xlwt
import os
import random
import time
import sys
# import mysql.connector
# import pymysql


class Logger(object):
    def __init__(self, filename="./Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def read_excel_CMNIST(path=''):
    # open label excel
    # workbook1 = xlrd.open_workbook(r".\2020-12-24-new-dataset-alltrain.xls")
    workbook1 = xlrd.open_workbook(path)
    # get all the sheets
    # print('process ', fold)
    print("path: ", path)
    sheet1_name = workbook1.sheet_names()[0]
    # get the sheet content by using sheet index
    sheet1 = workbook1.sheet_by_index(0)  # sheet index begins from [0]
    # sheet1 = workbook1.sheet_by_name('Sheet1')
    print('info of sheet :', sheet1.nrows-1)
    pathi = sheet1.col_values(0)[1:]
    Ei = sheet1.col_values(int(path[14]))[1:]
    Yi = sheet1.col_values(1)[1:]
    repeat_list_E = {
        "E0R_y0": 0,
        "E0R_y1": 0,
        "E0G_y0": 0,
        "E0G_y1": 0,
        "E0": 0,
        "E1R_y0": 0,
        "E1R_y1": 0,
        "E1G_y0": 0,
        "E1G_y1": 0,
        "E1": 0,
        "E2R_y0": 0,
        "E2R_y1": 0,
        "E2G_y0": 0,
        "E2G_y1": 0,
        "E2": 0,
        "E3R_y0": 0,
        "E3R_y1": 0,
        "E3G_y0": 0,
        "E3G_y1": 0,
        "E3": 0,
        "E4R_y0": 0,
        "E4R_y1": 0,
        "E4G_y0": 0,
        "E4G_y1": 0,
        "E4": 0,

    }
    repeat_list_noE = {
        "R_y0": 0,
        "R_y1": 0,
        "G_y0": 0,
        "G_y1": 0,
        "E": 0,

    }
    if "train" in path:
        for ii in range(len(Ei)):
            path_ = pathi[ii]
            img = Image.open(path_)
            img = np.asarray(img.convert('RGBA'))[:, :, :3]
            # print("R", np.max(img[:, :, 0]))
            # print("G", np.max(img[:, :, 1]))
            # print("B", np.max(img[:, :, 2]))
            keys1 = "E" + str(int(Ei[ii]))
            repeat_list_E[str(keys1)] += 1
            Rmax = np.max(img[:, :, 0])
            Gmax = np.max(img[:, :, 1])
            Bmax = np.max(img[:, :, 2])
            if Rmax > 128 and Yi[ii] == 0:
                keys2 = "E" + str(int(Ei[ii])) + "R_y0"
                repeat_list_E[str(keys2)] += 1
            if Rmax > 128 and Yi[ii] == 1:
                keys2 = "E" + str(int(Ei[ii])) + "R_y1"
                repeat_list_E[str(keys2)] += 1

            if Gmax > 128 and Yi[ii] == 0:
                keys2 = "E" + str(int(Ei[ii])) + "G_y0"
                repeat_list_E[str(keys2)] += 1
            if Gmax > 128 and Yi[ii] == 1:
                keys2 = "E" + str(int(Ei[ii])) + "G_y1"
                repeat_list_E[str(keys2)] += 1
        print("E", repeat_list_E)

    else:
        for ii in range(len(Ei)):
            path_ = pathi[ii]
            img = Image.open(path_)
            img = np.asarray(img.convert('RGBA'))[:, :, :3]
            # print("R", np.max(img[:, :, 0]))
            # print("G", np.max(img[:, :, 1]))
            # print("B", np.max(img[:, :, 2]))
            keys1 = "E"
            repeat_list_noE[str(keys1)] += 1
            Rmax = np.max(img[:, :, 0])
            Gmax = np.max(img[:, :, 1])
            Bmax = np.max(img[:, :, 2])
            if Rmax > 128 and Yi[ii] == 0:
                keys2 = "R_y0"
                repeat_list_noE[str(keys2)] += 1
            if Rmax > 128 and Yi[ii] == 1:
                keys2 = "R_y1"
                repeat_list_noE[str(keys2)] += 1

            if Gmax > 128 and Yi[ii] == 0:
                keys2 = "G_y0"
                repeat_list_noE[str(keys2)] += 1
            if Gmax > 128 and Yi[ii] == 1:
                keys2 = "G_y1"
                repeat_list_noE[str(keys2)] += 1
        print("noE", repeat_list_noE)

    # print(repeat_list/(sheet1.nrows-1))


dir_list = os.listdir(r"../Dataset_E")
for path in dir_list:
    if path[-1] == "s" and path[0] == "E" and "MNIST" in path:
        read_excel_CMNIST(path=os.path.join("../Dataset_E", path))
