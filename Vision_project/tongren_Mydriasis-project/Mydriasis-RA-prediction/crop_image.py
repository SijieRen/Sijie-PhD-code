import os
from PIL import Image
import time
import re

filter=[".jpg"] #设置过滤后的文件类型 当然可以设置多个类型

def all_path(dirname):

    result = []#所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):

        # print("1:",maindir) #当前主目录
        # print("2:",subdir) #当前主目录下的所有目录
        # print("3:",file_name_list)  #当前主目录下的所有文件

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)#合并成一个完整路径
            ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容

            if ext in filter:
                result.append(apath)

    return result


path_list = ['../data/JPG1600SIZE-95/Grade1','../data/JPG1600SIZE-95/Grade2','../data/JPG1600SIZE-95/Grade3',
             '../data/JPG1600SIZE-95/Grade4','../data/JPG1600SIZE-95/Grade5','../data/JPG1600SIZE-95/Grade6']
save_list = ['../data/JPG512SIZE-95/Grade1','../data/JPG512SIZE-95/Grade2','../data/JPG512SIZE-95/Grade3',
             '../data/JPG512SIZE-95/Grade4','../data/JPG512SIZE-95/Grade5','../data/JPG512SIZE-95/Grade6']
#path_list = ['/alidata1/PrimarySchool/Grade2']
#save_list = ["/alidata1/PNG1600SIZE/Grade2"]

for g in range(len(path_list)):
    start_time = time.time()
    file_name_list = all_path(path_list[g])
    print('Grade ',g+1 ,'is starting')
    print('num of stu in this grade is :', len(file_name_list))

    pic_time = start_time
    for i in range(len(file_name_list)):
        #len(file_name_list))

        img_path = file_name_list[i]
        img_open = Image.open(img_path)
        #img_crop = img_open.crop([490, 60, 2090, 1660])
        img_resize = img = img_open.resize((512, 512),Image.ANTIALIAS)
        if i%1000 == 0:
            print('{0} pictures have been resized!'.format(i))
            pictrans_time = time.time() - pic_time
            print('time of 1000 tans is : {}'.format(pictrans_time))
            pic_time = time.time()

        if not os.path.exists(save_list[g]):
            os.makedirs(save_list[g])
        #print(file_name_list[i]))
        img_resize.save(os.path.join(save_list[g],os.path.basename(file_name_list[i])),quality = 95)

    one_grade_time = time.time() - start_time
    print('grade {} time is {}'.format(g + 1, one_grade_time))

    print('Grade ', g+1 , 'is finishing')

