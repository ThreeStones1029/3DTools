'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-11-22 11:13:41
LastEditors: ShuaiLei
LastEditTime: 2023-11-22 15:17:55
'''
import numpy as np
import SimpleITK as sitk


class HausdorffDistance:
    def __init__(self, image1_path, image2_path, reset_coordinate_path, percentile=95):
        """
        parma, image1_path:第一例nii.gz路径
        param, image2_path:第二例nii.gz路径
        param, reset_path,重新装换坐标系后保存的路径
        param, percentile,hausdorff距离百分率
        """
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.reset_coordinate_path = reset_coordinate_path
        self.percentile = percentile

    def load_image(self, image_path):
        return sitk.ReadImage(image_path)

    def reset_coordinate(self, image1, image2, reset_path):
        
        origin=image1.GetOrigin() #这三句是获取的image1的位置和方向。
        spacing=image1.GetSpacing()
        direction=image1.GetDirection()
        image2.SetOrigin(origin) #将image2处理成和imaeg1一致的位置坐标系
        image2.SetSpacing(spacing)
        image2.SetDirection(direction)
        sitk.WriteImage(image2, reset_path) #转换坐标系后保存。

    def compute_hausdorff(self):

        image1 = self.load_image(self.image1_path)
        image2 = self.load_image(self.image2_path)

        self.reset_coordinate(image1, image2, self.reset_coordinate_path) # 统一坐标系
        # Get the boundary of the two images
        contour1 = sitk.LabelContour(image1)
        contour2 = sitk.LabelContour(image2)

        # Compute the distances from contour1 to contour2 and vice-versa
        distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(contour1, squaredDistance=False, useImageSpacing=True))
        distances_1_to_2 = sitk.Mask(distance_map, contour2)
        
        distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(contour2, squaredDistance=False, useImageSpacing=True))
        distances_2_to_1 = sitk.Mask(distance_map, contour1)

        # 获取两个距离图的数据,将两个数组合并
        all_distances = np.concatenate((sitk.GetArrayFromImage(distances_1_to_2).ravel(), sitk.GetArrayFromImage(distances_2_to_1).ravel()))
        
        # 过滤掉为0的距离值
        all_distances = all_distances[all_distances != 0]
        
        # 计算hausdorff距离的百分位数 0~100
        hausdorff = np.percentile(all_distances, self.percentile)
        return hausdorff


if __name__ == "__main__":
    gt_path = "tools/gt_and_pre/L5.nii.gz"
    pre_path = "tools/gt_and_pre/2_img_23_prediction_resampled.nii.gz"
    reset_coordinate_path = "tools/gt_and_pre/2_img_23_prediction_resampled_reset.nii.gz"

    hausdorff_distance = HausdorffDistance(gt_path, pre_path, reset_coordinate_path, percentile=100).compute_hausdorff()
    print(hausdorff_distance)
