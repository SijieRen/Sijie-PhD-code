import os
import SimpleITK as sitk
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image


special1_set = []
special2_set = []
# 图像重采样
def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0]):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    # 根据输出out_spacing设置新的size
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def read_nii_file(nii_file_path):
    sitk_data = sitk.ReadImage(nii_file_path)
    out_spacing = [1.0, 1.0, 1.0]  # 图像重采样，将out_spacing替换替换为期望输出 spacing:间距
    sitk_data = resample_image(sitk_data, out_spacing)
    return sitk.GetArrayFromImage(sitk_data), sitk_data  # 像素矩阵


def prepare_orimage(data_root_path, output_dir):  # 用与记录最大的mask宽与高，后期切图都用最大的宽、高
    for type_name in os.listdir(data_root_path):
        print("type_name:", type_name)
        output_dir_path = os.path.join(output_dir, type_name)
        os.makedirs(output_dir_path, exist_ok=True)
        type_name_path = os.path.join(data_root_path, type_name)
        origin_case_name_list = [case_name for case_name in os.listdir(type_name_path) if
                                 "_seg.nii.gz" not in case_name]
        num_files = len(origin_case_name_list)
        for origin_case_name in origin_case_name_list:
            print(f'processing {origin_case_name}')
            origin_case_name_path = os.path.join(type_name_path, origin_case_name)
            seg_case_name_path = os.path.join(type_name_path, origin_case_name.replace(".nii.gz", "_seg.nii.gz"))
            mask_image, origin_data = read_nii_file(seg_case_name_path)  
            z_list, y_list, x_list = np.nonzero(mask_image)  # 获取整个标注文件的所有非零点的坐标列表

            # get左上角  右下角 中心点位置
            min_h = np.min(y_list)
            min_w = np.min(x_list)
            max_h = np.max(y_list)
            max_w = np.max(x_list)
            min_d = np.min(z_list)
            max_d = np.max(z_list)
            center_h = int(max_h - (max_h - min_h) / 2)
            center_w = int(max_w - (max_w - min_w) / 2)
            center_d = int(max_d - (max_d - min_d) / 2)

            max_width, max_height, max_depth = 64, 64, 64
            width = np.max(x_list) - np.min(x_list)
            height = np.max(y_list) - np.min(y_list)
            depth = np.max(z_list) - np.min(z_list)
            if width > max_width: 
                max_width = width  # 如果当前这个nii文件中的最大宽度大于 max_width 则重新赋值
                special1_set.append(origin_case_name)
            if height > max_height: 
                max_height = height  
                special1_set.append(origin_case_name)
            if depth > max_depth: 
                max_depth = depth 
                special1_set.append(origin_case_name)

            origin_image, _ = read_nii_file(origin_case_name_path) 

            left_top_h = int(center_h - max_height / 2)  # 根据当前所有点的中心点来切图
            left_top_w = int(center_w - max_width / 2)
            first_depth = int(center_d - max_depth / 2 )
            right_bottom_h = int(center_h + max_height / 2)
            right_bottom_w = int(center_w + max_width / 2)
            end_depth = int(center_d + max_depth / 2)
            if first_depth < 0:
                first_depth = 0
                end_depth = 64
                crop_origin_img = origin_image[first_depth:end_depth, left_top_h:right_bottom_h, left_top_w:right_bottom_w]
                # todo: save origin nii
                output_save_name = f"{origin_case_name}"
                origin_out = sitk.GetImageFromArray(crop_origin_img.astype('int16'))  # 切图
                origin_out.SetOrigin(origin_data.GetOrigin())  # 坐标原点
                origin_out.SetSpacing(origin_data.GetSpacing())  # 像素间距
                # out.SetDirection(dicom_data.GetDirection())  # 方向
                prefix_output_save_name = output_save_name.split('_')
                new_output_save_name = f'{prefix_output_save_name [0]}_3D_64.nii.gz'
                origin_output_dicom_name = os.path.join(output_dir_path, new_output_save_name)
                sitk.WriteImage(origin_out, origin_output_dicom_name)
                print(f'{new_output_save_name} saved')
                num_files -= 1
                print(f'{num_files} dont saved')
                special2_set.append(origin_case_name)
            elif end_depth > origin_image.shape[0]:
                end_depth = origin_image.shape[0]
                first_depth = end_depth - 64
                crop_origin_img = origin_image[first_depth:end_depth, left_top_h:right_bottom_h, left_top_w:right_bottom_w]
                # save origin nii
                output_save_name = f"{origin_case_name}"
                origin_out = sitk.GetImageFromArray(crop_origin_img.astype('int16'))  # 切图
                origin_out.SetOrigin(origin_data.GetOrigin())  # 坐标原点
                origin_out.SetSpacing(origin_data.GetSpacing())  # 像素间距
                # out.SetDirection(dicom_data.GetDirection())  # 方向
                prefix_output_save_name = output_save_name.split('_')
                new_output_save_name = f'{prefix_output_save_name [0]}_3D_64.nii.gz'
                origin_output_dicom_name = os.path.join(output_dir_path, new_output_save_name)
                sitk.WriteImage(origin_out, origin_output_dicom_name)
                print(f'{new_output_save_name} saved')
                num_files -= 1
                print(f'{num_files} dont saved')
                special2_set.append(origin_case_name)     
            else:
                crop_origin_img = origin_image[first_depth:end_depth, left_top_h:right_bottom_h, left_top_w:right_bottom_w]
                # save origin nii
                output_save_name = f"{origin_case_name}"
                origin_out = sitk.GetImageFromArray(crop_origin_img.astype('int16'))  # 切图
                origin_out.SetOrigin(origin_data.GetOrigin())  # 坐标原点
                origin_out.SetSpacing(origin_data.GetSpacing())  # 像素间距
                # out.SetDirection(dicom_data.GetDirection())  # 方向
                prefix_output_save_name = output_save_name.split('_')
                new_output_save_name = f'{prefix_output_save_name [0]}_3D_64.nii.gz'
                origin_output_dicom_name = os.path.join(output_dir_path, new_output_save_name)
                sitk.WriteImage(origin_out, origin_output_dicom_name)
                print(f'{new_output_save_name} saved')
                num_files -= 1
                print(f'{num_files} dont saved')
    print("**************************************************************")
    print("special1_set:", special1_set)
    print('\n')
    print("special2_set:", special2_set)
    return None


if __name__ == '__main__':
    data_root_path = "../stas/zhongshan/zhongshan-hospital-STAS" 
    output_dir = "..Total_Stas_Data/stas_ZS_niiextract_noenhance_3D_64"
    prepare_orimage(data_root_path, output_dir)

