# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:00:09 2019

@author: landehuxi
"""
import os
import xlrd
import xlwt
from xlutils.copy import copy


def repeat(a, alist):
    repeat_num = 0
    for ii in range(len(alist)):
        if a == alist[ii]:
            repeat_num += 1

    return repeat_num


def read_excel():
    # open label excel
    workbook1 = xlrd.open_workbook(
        '/home/ruxin2/DailyLife/70025_650000(1)-副本.xls')
    # get all the sheets
    print(workbook1.sheet_names())
    sheet1_name = workbook1.sheet_names()[0]
    # get the sheet content by using sheet index
    sheet1 = workbook1.sheet_by_index(0)  # sheet index begins from [0]
    # sheet1 = workbook1.sheet_by_name('Sheet1')
    print(sheet1.name, sheet1.nrows, sheet1.ncols)
    global col_1_0
    col_1_0 = sheet1.col_values(0)
    print('len of col_1_0 is: ', len(col_1_0))

    id = sheet1.col_values(0)[2:]
    ip = sheet1.col_values(5)[2:]
    time = sheet1.col_values(7)[2:]

    return id, ip, time


# 同一个ip下30人以上投票的无效，同一秒内5个以上投票的无效。
ip_repeat = 30
time_repeat = 5


def writeExcel():
    # add an empty excel as oringinal one
    # , formatting_info=True)
    rb = xlrd.open_workbook('/home/ruxin2/DailyLife/70025_650000(1)-副本.xls')
    wb = copy(rb)
    ws = wb.get_sheet(0)
    row = 0
    id, ip, time = read_excel()
    ws.write(1, 9, "是否无效")
    ws.write(1, 10, "无效原因")
    print("开始处理ip无效")
    for i in range(len(ip)):
        i_repeat_num = repeat(ip[i], ip)
        if i_repeat_num >= 30:

            ws.write(i+2, 9, "无效票")
            ws.write(i+2, 10, "ip下超30人投票")
        else:
            ws.write(i+2, 9, "有效票")

    print("开始处理时间无效")
    for i in range(len(time)):
        i_repeat_num = repeat(time[i], time)
        if i_repeat_num >= 5:

            ws.write(i+2, 9, "无效票")
            ws.write(i+2, 10, "一秒内超5人投票")
    print("处理完毕，开始保存文件！")
    # TODO add _Optim to the end of the saving files name
    wb.save("/home/ruxin2/DailyLife/70025_650000(1)-副本-标记无效票.xls")


writeExcel()
