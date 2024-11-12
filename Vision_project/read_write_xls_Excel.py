# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:00:09 2019

@author: landehuxi
"""
import os
import xlrd
import xlwt
from xlutils.copy import copy

def read_excel():
    # open label excel
    workbook1 = xlrd.open_workbook('/alidata1/RA_Label_Index/Grade6_Ra_Index.xls')
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


# part of writting Excel
def writeExcel():
    # add an empty excel as oringinal one
    rb = xlrd.open_workbook('/alidata1/RA_Label_Index/Grade6_Ra_Index.xls')  # , formatting_info=True)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    row = 0
    for i in range(0, len(col_1_0)):
        ws.write(i, 0, (os.path.join('Grade6', os.path.basename(col_1_0[i]))))

    # TODO add _Optim to the end of the saving files name
    wb.save("/alidata1/RA_Label_Index/Grade6_Ra_Index_Optim.xls")


read_excel()
writeExcel()



