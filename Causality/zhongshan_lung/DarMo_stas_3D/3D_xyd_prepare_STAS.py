import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
# 生成实验所用的32*32结节图片npy数据
import numpy as np
from os import path
import pandas as pd
import os
from matplotlib import pyplot as plt
from PIL import Image
from collections import Counter
import scipy
from scipy import ndimage
from glob import glob

import SimpleITK as sitk

window_center = -600
window_width = 1200

def adjustGrayImage_stdMean(img):
        mean = np.mean(img)
        std = np.std(img)
        eps = 1e-6
        return (img - mean) / (std + eps)

def resample(image, spacing, new_spacing=[1, 1, 1], order=1):  # [0.5, 0.5, 0.5]
    # shape can only be int, so has to be rounded.
    new_shape = np.round(image.shape * spacing / new_spacing)
    # the actual spacing to resample.
    resample_spacing = spacing * image.shape / new_shape
    resize_factor = new_shape / image.shape
    image_new = scipy.ndimage.interpolation.zoom(image, resize_factor, mode='nearest', order=order)
    # if image.shape[1] == 1024:
    #     image_new = scipy.ndimage.interpolation.zoom(image_new, [1, 0.5, 0.5], mode='nearest', order=order)
    return image_new, resample_spacing, resize_factor

def get_basic_inf(path, pathofmask):
    # 读取数据
    image = sitk.ReadImage(path)
    image_mask = sitk.ReadImage(pathofmask)

    spacing = np.array(image.GetSpacing())
    spacing = [spacing[2], spacing[1], spacing[0]]
    spacing = np.array(spacing)

    spacing1 = np.array(image.GetSpacing())
    spacing1 = [spacing1[2], spacing1[1], spacing1[0]]
    spacing1 = np.array(spacing1)

    image = adjustGrayImage_stdMean(sitk.GetArrayFromImage(image))


    # convert to ndarry# 输出数据shape
    # 将CT矩阵从[Z,Y,X]换至[X,Y,Z],并读取各自spacing (TT:未换，为了适应后面resample中[Z,Y,X]的顺序)
    arrayofCT = np.array(image)  # 使用GetArrayFromImage时，X,Z位置调换，输出形状为：(Depth, Height, Width)即（Z,Y,X）。
    
    # 将结节矩阵从[Z,Y,X]换至[X,Y,Z]
    arrayofMask = np.array(sitk.GetArrayFromImage(image_mask))
    

    # spacexy = spacing[1]
    # if spacing[2] < spacing[1]:
    #     spacexy = spacing[2]
    # 保证xy方向不缩小，以免丢失细节，影响征象判断，同时对z方向resample。
    # resize_factor = (spacexy, spacexy, spacexy)
    arrayofCT, resample_spacing, _ = resample(arrayofCT, spacing) # Z Y X
    arrayofMask, resample_spacing1, _ = resample(arrayofMask, spacing1)

    # 取出非零部分（即结节）
    arrayofLNLo = np.array(np.nonzero(arrayofMask))  # 返回非零元素的位置,返回矩阵形状为3*（非零元素个数），每行分别代表一个维度的坐标
    # print("shape.", arrayofLNLo.shape)
    # print(np.min(arrayofLNLo, axis=0), np.max(arrayofLNLo, axis=0))
    print(np.min(arrayofLNLo, axis=1), np.max(arrayofLNLo, axis=1))
    ZYX_chazhi = np.max(arrayofLNLo, axis=1) - np.min(arrayofLNLo, axis=1)
    print("ZYX_chazhi: ", ZYX_chazhi) # Z Y X
    # print(np.min(arrayofLNLo, axis=2), np.max(arrayofLNLo, axis=2))
    ZCo = list(arrayofLNLo[0])
    YCo = list(arrayofLNLo[1])
    XCo = list(arrayofLNLo[2])
    # print("X array, Y array, Z array",XCo,YCo,ZCo)
    # arrayofLNCoor = [XCo, YCo, ZCo]
    # arrayofLNCoor = np.array(arrayofLNCoor).T  # 将非零元素位置转换为标准的三维坐标形式 ZYX
    # print('the shape of nodule array is %d rows and %d columns' % (arrayofLNCoor.shape[0], arrayofLNCoor.shape[1]))
    # 取所有结节的中心点作为结节中心
    centerZYX = ((np.min(arrayofLNLo, axis=1) + np.max(arrayofLNLo, axis=1))/2).astype(int)  # 返回整个结节的中心点
    print("center coordinate", centerZYX)
    # print('the center of nodule(calculated using all points) is:%s' % center)
    return arrayofCT, arrayofMask, centerZYX, ZYX_chazhi


def img_display_and_save(arrayofCT, centerZYX, count_saveimg, save_path, ZYX_chazhi):

    level = -600
    window = 1200
    min = (2 * level - window) / 2.0 + 0.5
    max = (2 * level + window) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)
    # sizeofcuts = 48 * 2
    sizeofcuts = []
    for ii in range(len(ZYX_chazhi)):
        if ZYX_chazhi[ii] % 2 == 0:
            sizeofcuts.append(ZYX_chazhi[ii])
        else:
            sizeofcuts.append(ZYX_chazhi[ii]+1)


    # centerX, centerY, MaxLayerIndex = centerXYZ
    
    # 横截面
    # 此处提取64*64*64 cube
    max_mask_coordinate = np.max(ZYX_chazhi)
    if max_mask_coordinate %2 ==0:
        max_mask_coordinate = max_mask_coordinate
    else:
        max_mask_coordinate += 1
    background2 = np.zeros((max_mask_coordinate, max_mask_coordinate, max_mask_coordinate), dtype=np.uint8)  # 用作截取结节图像的背景 Z Y X
    img = arrayofCT
    # img = img - min
    # img = np.trunc(img * dFactor)
    # img[img < 0.0] = 0
    # img[img > 255.0] = 255
    # row Z, col Y, hei Z
    row0 = int(centerZYX[0] - sizeofcuts[0]/ 2)  # 因为对应到img中row为高度，所以，这里取centerXYZ[2]
    row1 = int(centerZYX[0] + sizeofcuts[0]/ 2)
    column0 = int(centerZYX[1] - sizeofcuts[1] / 2)
    column1 = int(centerZYX[1] + sizeofcuts[1] / 2)
    height0 = int(centerZYX[2] - sizeofcuts[2] / 2)
    height1 = int(centerZYX[2] + sizeofcuts[2] / 2)
    if row0 < 0:
        row0 = 0
    if column0 < 0:
        column0 = 0
    if height0 < 0:
        height0 = 0
    if row1 > arrayofCT.shape[0]:  # 数据集中有部分数据只有40-60个slice，在裁剪时row会超出上下界
        row1 = arrayofCT.shape[0]
    if column1 > arrayofCT.shape[1]:
        column1 = arrayofCT.shape[1]
    if height1 > arrayofCT.shape[2]:
        height1 = arrayofCT.shape[2]
    cubecut = img[row0:row1, column0:column1, height0: height1]# Z Y X  # 注意参数顺序，前面的表示行，后面的表示列,高，对应的是坐标位置，不要和坐标值搞混淆；且使用：时注意，1：n表示从1到n，不包括n
    meanofimgcut2 = int(np.array(cubecut).mean())  # 对截取图像像素取平均
    # background2[:, :, :] = meanofimgcut2 # padding均值灰度  #注释掉 padding0
    background2[(sizeofcuts[0] - (row1 - row0)): sizeofcuts[0], 
                (sizeofcuts[1] - (column1 - column0)): sizeofcuts[1], 
                (sizeofcuts[2] - (height1 - height0)): sizeofcuts[2]] = np.uint8(
        cubecut[:, :, :])  # 用所截取图像来填充背景，若截取图像小于背景图大小，对于空出部分，已用平均灰度/0填充
    # np.save(path + str(count_saveimg) + '.npy', background2)
    np.save(save_path, background2)
    print(f'{count_saveimg} saved---------')

def read_nii_file(nii_file_path):
    sitk_data = sitk.ReadImage(nii_file_path)
    return sitk.GetArrayFromImage(sitk_data), sitk_data 


def prepare_orimage(data_root_path, output_dir, if_save=False): 
    print("save in :", output_dir) 
    for type_name in os.listdir(data_root_path):
        type_name_path = os.path.join(data_root_path, type_name)
        output_path = os.path.join(output_dir, type_name)
        os.makedirs(output_path, exist_ok=True)
        origin_case_name_list = [case_name for case_name in os.listdir(type_name_path) if#中山医院_mask 肿瘤医院_seg
                                 "_seg.nii.gz" not in case_name] #TODO 对不同数据是否需要修改处理后缀
        mask_case_name_list = [case_name for case_name in os.listdir(type_name_path) if #for 中山医院
                                 "_seg.nii.gz" in case_name] #TODO 对不同数据是否需要修改处理后缀
        save_p = 0
        for origin_case_name in origin_case_name_list:
            origin_case_name_path = os.path.join(type_name_path, origin_case_name)
            # seg_case_name_path = os.path.join(type_name_path, origin_case_name.replace(".nii.gz", "_seg.nii.gz"))#for 肿瘤医院
            
            for mask_name in mask_case_name_list:
                if origin_case_name[:-7] in mask_name: # XXX.nii.gz for origin_case_name
                    mask_case_name = mask_name
                    # break
                    seg_case_name_path = os.path.join(type_name_path, mask_case_name)# for 中山医院 and 肿瘤医院
                    print("99999999",seg_case_name_path)
                    mask_image, _ = read_nii_file(seg_case_name_path)  
                    # arrayofCT, arrayofMask, centerXYZ = get_basic_inf(CT_file_path, mask_file_path)
                    # img_display_and_save(arrayofCT, centerXYZ, key_id[index], path_savecuts)

                    arrayofCT, arrayofMask, centerZYX, max_min = get_basic_inf(origin_case_name_path, seg_case_name_path)
                    origin_prefix_name = origin_case_name[:-7]##中山[:-7]  肿瘤[:-7]
                    output_image_dir = os.path.join(output_path, origin_prefix_name)
                    #print("origin_prefix_name",origin_prefix_name)
                    os.makedirs(output_image_dir, exist_ok=True)
                    output_save_name = f"{str(save_p + 1).zfill(3)}.npy" #同一个人如果有多张图片是否在不同depth
                    origin_output_image_name = os.path.join(output_image_dir, output_save_name)
                    #print("output_image_dir", output_image_dir)
                    #print("SAVE SAVE", origin_output_image_name)
                    # origin_output_image.save(origin_output_image_name)

                    if if_save:
                        img_display_and_save(arrayofCT, centerZYX, save_p, origin_output_image_name, max_min)
                        print("save num:", save_p)
                    else:
                        print("do not save {}, only print".format(save_p))

                    

                    save_p+=1



                    # z_list, y_list, x_list = np.nonzero(mask_image) 
                    # min_h = np.min(y_list)
                    # min_w = np.min(x_list)
                    # max_h = np.max(y_list)
                    # max_w = np.max(x_list)
                    # center_h = int(max_h - (max_h - min_h) / 2)
                    # center_w = int(max_w - (max_w - min_w) / 2)

                    # max_width, max_height = 64, 64
                    # width = np.max(x_list) - np.min(x_list)
                    # height = np.max(y_list) - np.min(y_list)
                    # if width > max_width: max_width = width  
                    # if height > max_height: max_height = height  

                    # origin_image, _ = read_nii_file(origin_case_name_path) 
                    # max_area_depth = [0, 0]

                    # for depth in set(z_list):
                    #     mask_single = mask_image[depth, :, :]
                    #     area = mask_single[mask_single > 0].sum()
                    #     if area > max_area_depth[1]:
                    #         max_area_depth[0] = depth
                    #         max_area_depth[1] = area
                    # # 原始按照mask区域最大部分提取
                    # # origin_single = origin_image[max_area_depth[0], :, :]
                    # # 增加mask数据 提取
                    # origin_single = np.where(mask_image[max_area_depth[0], :, :] > 0, origin_image[max_area_depth[0], :, :], 1)

                    # left_top_h = max(int(center_h - max_height / 2), 0)
                    # left_top_w = max(int(center_w - max_width / 2), 0)
                    # right_bottom_h = min(int(center_h + max_height / 2), mask_single.shape[0])
                    # right_bottom_w = min(int(center_w + max_width / 2), mask_single.shape[1])
                    # crop_origin_img = origin_single[left_top_h:right_bottom_h, left_top_w:right_bottom_w]
                    # # print("111",origin_case_name)
                    # # print("222",origin_case_name.rsplit("_", 2))
                    # origin_prefix_name = origin_case_name[:-7]##中山[:-7]  肿瘤[:-7]
                    # if crop_origin_img.shape != (64,64):
                    #     print(origin_prefix_name)
                    # output_image_dir = os.path.join(output_path, origin_prefix_name)
                    # #print("origin_prefix_name",origin_prefix_name)
                    # os.makedirs(output_image_dir, exist_ok=True)
                    # output_save_name = f"{str(depth + 1).zfill(3)}_{str(save_p + 1).zfill(3)}.png" #同一个人如果有多张图片是否在不同depth
                    # min_window = (2*window_center - window_width)/2.0 + 0.5
                    # max_window = (2*window_center + window_width)/2.0 + 0.5
                    # dFactor = 255.0 / (max_window - min_window)
                    # window_single_processing = (crop_origin_img - min_window) * dFactor
                    # window_single_processing[window_single_processing < 0] = 0      
                    # window_single_processing[window_single_processing > 255] = 255
                    # origin_output_image = Image.fromarray(window_single_processing.astype(np.uint8))
                    # origin_output_image_name = os.path.join(output_image_dir, output_save_name)
                    # #print("output_image_dir", output_image_dir)
                    # #print("SAVE SAVE", origin_output_image_name)
                    # origin_output_image.save(origin_output_image_name)
                    # save_p+=1
                    # print("save num:", save_p)
    return None


if __name__ == '__main__':
    data_root_path = "../../../../data/zhongshan_Lung_data/肿瘤医院/" 
    output_dir = "../../../../data/Total_Stas_unzip_zhongshan_Lung_data_3D-bgMAX_maskXYZgezi_padding0_stdMean_spacing111_old-check/肿瘤医院/"
    if_save = 1
    prepare_orimage(data_root_path, output_dir, if_save)

