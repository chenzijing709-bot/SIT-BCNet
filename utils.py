import os
import time

import pandas as pd
import torch
import random
import logging
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
import xarray as xr
from global_land_mask import globe


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def visualize(inputs, targets, outputs, epoch, idx, cache_dir):
    _, axarray = plt.subplots(3, targets.shape[1], figsize=(targets.shape[1] * 5, 10))

    for t in range(targets.shape[1]):
        axarray[0][t].imshow(inputs[0, t, :,:].detach().cpu().numpy(), cmap='gray')
        axarray[1][t].imshow(targets[0, t, :,:].detach().cpu().numpy(), cmap='gray')
        axarray[2][t].imshow(outputs[t, :,:].detach().cpu().numpy(), cmap='gray')

    plt.savefig(os.path.join(cache_dir, '{:03d}-{:03d}.png'.format(epoch, idx)))
    plt.close()

def plot_loss(loss_records, loss_type, epoch, plot_dir, step):
    plt.plot(range((epoch + 1) // step), loss_records, label=loss_type)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, '{}_loss_records.png'.format(loss_type)))
    plt.close()

# 计算 R² (决定系数)
def compute_r2(predictions, targets, mask):
    predictions = predictions * mask
    targets = targets * mask

    # 获取数据的形状 (batch_size, Seq_len, height, width)
    batch_size = predictions.shape[0]
    Seq_len = predictions.shape[1]

    r2 = 0
    excel_filename = "E:\code\Prediction_SIT\\result\index\\r220222023111222fff.xlsx"

    # 对每个 batch 和每个图像帧进行计算
    for batch in range(batch_size):
        for frame in range(Seq_len):
            # 获取当前帧的预测值和目标值
            pred_frame = predictions[batch, frame].flatten()  # 拉平成一维
            target_frame = targets[batch, frame].flatten()  # 拉平成一维

            # 计算目标的均值
            target_mean = np.mean(target_frame)

            # 计算总平方和 (TSS)
            tss = np.sum((target_frame - target_mean) ** 2)

            # 计算残差平方和 (RSS)
            rss = np.sum((target_frame - pred_frame) ** 2)

            # 计算当前帧的 R²
            r2_frame = 1 - (rss / tss)


            # df_mae = pd.DataFrame([r2_frame], columns=[f'R2_{frame + 1}'])
            # if os.path.exists(excel_filename):
            #     # 如果文件存在，读取当前文件并追加新的数据
            #     with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            #         df_mae.to_excel(writer, index=False, header=False, startcol=0,
            #                         startrow=writer.sheets['Sheet1'].max_row)
            # else:
            #     # 如果文件不存在，创建新的文件并写入
            #     df_mae.to_excel(excel_filename, index=False)

            r2 += r2_frame

    # 计算所有帧的平均 R²
    r2 /= (batch_size * Seq_len)
    return r2
def MAE2(pred, true, mask_land):
    # 计算每个预测值和真实值的绝对差
    abs_errors = np.abs(pred - true)
    # 加上掩码
    abs_errors = abs_errors * mask_land
    # 创建一个布尔数组，只有当真实值大于2时才为True
    mask = true > 2
    mask = mask.squeeze()
    abs_errors = abs_errors.squeeze()

    mae_per_image_list = []
    excel_filename = "E:\code\Prediction_SIT\\result\index\mae220222023111222fff.xlsx"
    for i in range(mask.shape[0]):
        # 使用这个数字索引来只选择那些真实值大于2的误差
        abs_errors_masked = abs_errors[i][mask[i]]
        # 对每张图片独立求MAE，然后求这些MAE的均值
        mae_per_image = np.mean(abs_errors_masked)
        # df_mae = pd.DataFrame([mae_per_image], columns=[f'MAE_{i + 1}'])
        # if os.path.exists(excel_filename):
        #     # 如果文件存在，读取当前文件并追加新的数据
        #     with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        #         df_mae.to_excel(writer, index=False, header=False, startcol=0, startrow=writer.sheets['Sheet1'].max_row)
        # else:
        #     # 如果文件不存在，创建新的文件并写入
        #     df_mae.to_excel(excel_filename, index=False)

        mae_per_image_list.append(mae_per_image)
    # 得到所有图片的MAE平均值
    mean_mae = np.mean(mae_per_image_list)
    return mean_mae

def MAE(pred, true, mask):
    #计算每个预测值和真实值的绝对差
    abs_errors = np.abs(pred - true)
    # 加上掩码
    abs_errors = abs_errors * mask
    mae_per_image = np.sum(abs_errors, axis=(2, 3)) / np.sum(mask, axis=(0, 1))  # 按照有效像素计算均值
    # 对每张图片独立求MAE，然后求这些MAE的均值
    # 这里，我们仍然按照pred的4D结构，保持图片维度不变，仅在图像像素上求均值
    #mae_per_image = np.mean(abs_errors, axis=(2, 3))

    excel_filename = "E:\code\Prediction_SIT\\result\index\mae220222023111222fff.xlsx"
    # 创建一个DataFrame保存每张图片的MAE
    # df_mae = pd.DataFrame(mae_per_image, columns=[f'Image_{i + 1}' for i in range(mae_per_image.shape[1])]).T
    #
    # # 检查文件是否存在，如果存在则追加到现有文件
    # if os.path.exists(excel_filename):
    #     # 如果文件存在，读取当前文件并追加新的数据
    #     with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    #         df_mae.to_excel(writer, index=False, header=False, startcol=0, startrow=writer.sheets['Sheet1'].max_row)
    # else:
    #     # 如果文件不存在，创建新的文件并写入
    #     df_mae.to_excel(excel_filename, index=False)
    # 得到所有图片的MAE平均值
    mean_mae = np.mean(mae_per_image)
    return mean_mae

def MSE(pred, true,mask):
    # 计算每个预测值和真实值差的平方
    squared_errors = (pred - true) ** 2

    # 使用掩码来屏蔽掉陆地区域 (mask == 0)
    squared_errors = squared_errors * mask  # 只有 mask == 1 的地方会保留值，mask == 0 的地方会变成 0

    # 对每张图片独立求均方误差，轴 (2, 3) 表示图像的纬度和经度，按海洋区域计算
    mse_per_image = np.sum(squared_errors, axis=(2, 3)) / np.sum(mask, axis=(0, 1))  # 按照有效像素计算均值

    # 对每张图片独立求均方误差，然后求这些MSE的均值
    # 这里，我们仍然按照pred的4D结构，保持图片维度不变，仅在图像像素上求均值
    #mse_per_image = np.mean(squared_errors, axis=(2, 3))

    excel_filename = "E:\code\Prediction_SIT\\result\index\mse220222023111222fff.xlsx"
    # 创建一个DataFrame保存每张图片的MAE
    # df_mae = pd.DataFrame(mse_per_image, columns=[f'Image_{i + 1}' for i in range(mse_per_image.shape[1])]).T
    #
    # # 检查文件是否存在，如果存在则追加到现有文件
    # if os.path.exists(excel_filename):
    #     # 如果文件存在，读取当前文件并追加新的数据
    #     with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    #         df_mae.to_excel(writer, index=False, header=False, startcol=0, startrow=writer.sheets['Sheet1'].max_row)
    # else:
    #     # 如果文件不存在，创建新的文件并写入
    #     df_mae.to_excel(excel_filename, index=False)

    # 得到所有图片的MSE平均值
    mean_mse = np.mean(mse_per_image)
    return mean_mse
    #return np.mean((pred - true) ** 2, axis=(0, 1)).sum()

# cite the 'PSNR' code from E3D-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py line 39-40
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255) - np.uint8(true * 255)) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def mask_land():
    file_path = r'E:\SeaIceDataSet\PIOMAS\codeTOnc\piomas_bin_reader-master\data\PIOMAS\PIOMAS_2011-2021_summer_7days_WithoutLand\avg_thickness_data_20110501_to_20110507.nc'
    data = xr.open_dataset(file_path)
    latitude = data['latitude']  # 获取纬度数据
    longitude = data['longitude']  # 获取经度数据

    # 提取实际的经纬度值
    lats = latitude.values
    lons = longitude.values
    lons[lons > 180] = lons[lons > 180] - 360

    # 获取纬度和经度的网格形状
    num_lat, num_lon = lats.shape
    mask = np.full((num_lat, num_lon), np.nan)
    mask2 = np.full((num_lat, num_lon), np.nan)

    for i in range(num_lat):
        for j in range(num_lon):
            lat = lats[i, j]
            lon = lons[i, j]
            if globe.is_ocean(lat, lon):
                mask[i, j] = 1  # 海洋区域标记为 1
                mask2[i, j] = 1
            else:
                mask[i, j] = 0  # 陆地区域标记为 0
                mask2[i,j] = 0

    for i in range(1, num_lat - 1):  # 避免边界
        for j in range(1, num_lon - 1):  # 避免边界
            if mask[i, j] == 0:
                if np.any(mask[i - 1:i + 2, j - 1:j + 2] == 1):  # 在3x3邻域内检测是否有海洋
                    mask2[i, j] = 1

    return mask2

def mask_laptev_sea():
    """
    生成拉普捷夫海专属掩码：
    - 1：拉普捷夫海区域（海洋+经纬度范围）
    - 0：非拉普捷夫海区域（陆地/其他海洋）
    """
    # 1. 读取PIOMAS网格数据（复用原文件路径，保证网格一致）
    file_path = r'E:\SeaIceDataSet\PIOMAS\codeTOnc\piomas_bin_reader-master\data\PIOMAS\PIOMAS_2011-2021_summer_7days_WithoutLand\avg_thickness_data_20110501_to_20110507.nc'
    data = xr.open_dataset(file_path)
    latitude = data['latitude']
    longitude = data['longitude']

    # 2. 处理经纬度（统一经度范围为-180~180，兼容原逻辑）
    lats = latitude.values  # 形状 [num_lat, num_lon]
    lons = longitude.values
    lons[lons > 180] = lons[lons > 180] - 360  # 把0~360转为-180~180

    # 3. 初始化基础掩码（海洋/陆地）
    num_lat, num_lon = lats.shape
    mask_land = np.full((num_lat, num_lon), 0)  # 0=陆地，1=海洋
    for i in range(num_lat):
        for j in range(num_lon):
            lat = lats[i, j]
            lon = lons[i, j]
            if globe.is_ocean(lat, lon):  # 保留原海洋判断逻辑
                mask_land[i, j] = 1

    # 4. 定义拉普捷夫海经纬度范围（核心：锁定目标区域）
    # 拉普捷夫海标准范围：70°N ~ 85°N，100°E ~ 140°E（转换为-180~180则为100~140）
    lat_min, lat_max = 70.0, 85.0
    lon_min, lon_max = 100.0, 180.0

    # 5. 生成拉普捷夫海初始掩码（经纬度筛选+海洋约束）
    mask_laptev = np.zeros((num_lat, num_lon))
    for i in range(num_lat):
        for j in range(num_lon):
            lat = lats[i, j]
            lon = lons[i, j]
            # 条件：在拉普捷夫海经纬度范围内 + 是海洋区域
            if (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max) and (mask_land[i, j] == 1):
                mask_laptev[i, j] = 1

    # 6. 邻域修正（填补边缘微小缺口，复用原mask_land的邻域逻辑）
    mask_laptev_fixed = mask_laptev.copy()
    for i in range(1, num_lat - 1):
        for j in range(1, num_lon - 1):
            if mask_laptev[i, j] == 0:
                # 若3x3邻域内有拉普捷夫海区域，也标记为1（填补边缘）
                if np.any(mask_laptev[i-1:i+2, j-1:j+2] == 1):
                    mask_laptev_fixed[i, j] = 1

    return mask_laptev_fixed
def mask_three_regions():
    """
    生成三个目标敏感区区域的掩膜：
    - 0 = 非敏感区
    - 1 = Central Arctic (83°N–88°N, 120°E–80°W)
    - 2 = ESS–Laptev–Chukchi (70°N–75°N, 105°E–180°E & -180°W)
    - 3 = Greenland–Barents (80°N–84°N, 40°W–40°E)
    """

    # ---------------------------
    # 1. 读取 PIOMAS 网格
    # ---------------------------
    file_path = r'E:\SeaIceDataSet\PIOMAS\codeTOnc\piomas_bin_reader-master\data\PIOMAS\PIOMAS_2011-2021_summer_7days_WithoutLand\avg_thickness_data_20110501_to_20110507.nc'
    ds = xr.open_dataset(file_path)

    lats = ds['latitude'].values
    lons = ds['longitude'].values.astype(float)

    # 转换经度至 -180~180
    lons[lons > 180] = lons[lons > 180] - 360

    num_lat, num_lon = lats.shape

    # ---------------------------
    # 2. 海陆掩膜
    # ---------------------------
    mask_land = np.zeros((num_lat, num_lon), dtype=int)
    for i in range(num_lat):
        for j in range(num_lon):
            if globe.is_ocean(lats[i, j], lons[i, j]):
                mask_land[i, j] = 1  # 海洋 = 1

    # 最终掩膜（0 = 非敏感区）
    mask = np.zeros((num_lat, num_lon), dtype=int)

    # ====================================================
    # 3. 三个敏感区定义
    # ====================================================

    # Helper：判断 lon 范围（支持跨180°）
    def lon_in_range(lon, lon_min, lon_max):
        if lon_min <= lon_max:
            return lon_min <= lon <= lon_max
        else:
            # 跨180°
            return (lon >= lon_min) or (lon <= lon_max)

    # ---------------------------
    # (1) Central Arctic -> 1
    # 83°N–88°N, 120°E–80°W  → lon: 120°E ~ -80°
    # ---------------------------
    for i in range(num_lat):
        for j in range(num_lon):
            lat = lats[i, j]
            lon = lons[i, j]
            if (83 <= lat <= 88) and lon_in_range(lon, 120, -80) and (mask_land[i, j] == 1):
                mask[i, j] = 1

    # ---------------------------
    # (2) ESS–Laptev–Chukchi -> 2
    # 70°N–75°N, 105°E – 180°E + -180° ~ （跨180°）
    # ---------------------------
    for i in range(num_lat):
        for j in range(num_lon):
            lat = lats[i, j]
            lon = lons[i, j]
            if (70 <= lat <= 75) and lon_in_range(lon, 105, -180) and (mask_land[i, j] == 1):
                mask[i, j] = 2

    # ---------------------------
    # (3) Greenland–Barents -> 3
    # 80°N–84°N, -40°W – 40°E
    # ---------------------------
    for i in range(num_lat):
        for j in range(num_lon):
            lat = lats[i, j]
            lon = lons[i, j]
            if (80 <= lat <= 84) and lon_in_range(lon, -40, 40) and (mask_land[i, j] == 1):
                mask[i, j] = 3

    # ---------------------------
    # 4. 邻域修正（填补边界）
    # ---------------------------
    mask_fixed = mask.copy()
    for i in range(1, num_lat - 1):
        for j in range(1, num_lon - 1):
            if mask[i, j] == 0:
                if np.any(mask[i - 1:i + 2, j - 1:j + 2] > 0):
                    mask_fixed[i, j] = mask[i - 1:i + 2, j - 1:j + 2].max()

    return mask_fixed

def mask_two_regions():
    """
    生成2个目标敏感区区域的掩膜：
    - 0 = 非敏感区
    - 1 = 中心环带 (80°N–85°N, 180°E到180°W)
    - 2 = Laptev–Chukchi (72°N–77°N, -180°E到-135°W)
    """

    # ---------------------------
    # 1. 读取 PIOMAS 网格
    # ---------------------------
    file_path = r'E:\SeaIceDataSet\PIOMAS\codeTOnc\piomas_bin_reader-master\data\PIOMAS\PIOMAS_2011-2021_summer_7days_WithoutLand\avg_thickness_data_20110501_to_20110507.nc'
    ds = xr.open_dataset(file_path)

    lats = ds['latitude'].values
    lons = ds['longitude'].values.astype(float)

    # 转换经度至 -180~180
    lons[lons > 180] = lons[lons > 180] - 360

    num_lat, num_lon = lats.shape

    # ---------------------------
    # 2. 海陆掩膜
    # ---------------------------
    mask_land = np.zeros((num_lat, num_lon), dtype=int)
    for i in range(num_lat):
        for j in range(num_lon):
            if globe.is_ocean(lats[i, j], lons[i, j]):
                mask_land[i, j] = 1  # 海洋 = 1

    # 最终掩膜（0 = 非敏感区）
    mask = np.zeros((num_lat, num_lon), dtype=int)

    # ====================================================
    # 3. 三个敏感区定义
    # ====================================================

    # Helper：判断 lon 范围（支持跨180°）
    def lon_in_range(lon, lon_min, lon_max):
        if lon_min <= lon_max:
            return lon_min <= lon <= lon_max
        else:
            # 跨180°
            return (lon >= lon_min) or (lon <= lon_max)

    # ---------------------------
    # (1) Central Arctic -> 1  (80°N–85°N, 180°E到180°W)
    # 83°N–88°N, 120°E–80°W  → lon: 120°E ~ -80°
    # ---------------------------
    for i in range(num_lat):
        for j in range(num_lon):
            lat = lats[i, j]
            lon = lons[i, j]
            if (80 <= lat <= 85) and (mask_land[i, j] == 1):
                mask[i, j] = 1

    # ---------------------------
    # (2) Laptev–Chukchi -> 2 (72°N–77°N, -180°E到-135°W)
    # 70°N–75°N, 105°E – 180°E + -180° ~ （跨180°）
    # ---------------------------
    for i in range(num_lat):
        for j in range(num_lon):
            lat = lats[i, j]
            lon = lons[i, j]
            if (72 <= lat <= 77) and lon_in_range(lon, -180, -135) and (mask_land[i, j] == 1):
                mask[i, j] = 2


    # ---------------------------
    # 4. 邻域修正（填补边界）
    # ---------------------------
    mask_fixed = mask.copy()
    for i in range(1, num_lat - 1):
        for j in range(1, num_lon - 1):
            if mask[i, j] == 0:
                if np.any(mask[i - 1:i + 2, j - 1:j + 2] > 0):
                    mask_fixed[i, j] = mask[i - 1:i + 2, j - 1:j + 2].max()

    return mask_fixed
def mask_random_two_regions():
    """
    生成2个随机敏感区（块状连续）：
    - 0 = 非敏感区
    - 1 = 随机区域1（面积≈Central Arctic）
    - 2 = 随机区域2（面积≈Laptev–Chukchi）
    """

    import numpy as np
    import xarray as xr
    import random
    from collections import deque
    from global_land_mask import globe

    # ---------------------------
    # 1. 读取 PIOMAS 网格
    # ---------------------------
    file_path = r'E:\SeaIceDataSet\PIOMAS\codeTOnc\piomas_bin_reader-master\data\PIOMAS\PIOMAS_2011-2021_summer_7days_WithoutLand\avg_thickness_data_20110501_to_20110507.nc'
    ds = xr.open_dataset(file_path)

    lats = ds['latitude'].values
    lons = ds['longitude'].values.astype(float)

    lons[lons > 180] -= 360

    num_lat, num_lon = lats.shape

    # ---------------------------
    # 2. 海洋mask
    # ---------------------------
    mask_land = np.zeros((num_lat, num_lon), dtype=int)
    for i in range(num_lat):
        for j in range(num_lon):
            if globe.is_ocean(lats[i, j], lons[i, j]):
                mask_land[i, j] = 1

    # ---------------------------
    # 3. 获取真实敏感区面积（用于匹配）
    # ---------------------------
    true_mask = mask_two_regions()
    area1 = np.sum(true_mask == 1)
    area2 = np.sum(true_mask == 2)

    # ---------------------------
    # 4. 区域生长函数（核心）
    # ---------------------------
    def grow_region(target_area):
        visited = np.zeros((num_lat, num_lon), dtype=bool)

        # 找随机海洋点
        ocean_points = np.argwhere(mask_land == 1)
        seed = tuple(ocean_points[random.randint(0, len(ocean_points) - 1)])

        region = set([seed])
        queue = deque([seed])
        visited[seed] = True

        while len(region) < target_area and queue:
            i, j = queue.popleft()

            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i + di, j + dj

                if 0 <= ni < num_lat and 0 <= nj < num_lon:
                    if mask_land[ni, nj] == 1 and not visited[ni, nj]:
                        visited[ni, nj] = True
                        region.add((ni, nj))
                        queue.append((ni, nj))

                        if len(region) >= target_area:
                            break

        return region

    # ---------------------------
    # 5. 生成两个随机区域
    # ---------------------------
    region1 = grow_region(area1)
    region2 = grow_region(area2)

    # ---------------------------
    # 6. 构建mask
    # ---------------------------
    mask = np.zeros((num_lat, num_lon), dtype=int)

    for (i, j) in region1:
        mask[i, j] = 1

    for (i, j) in region2:
        mask[i, j] = 2

    # ---------------------------
    # 7. 邻域修正（与你原代码一致）
    # ---------------------------
    mask_fixed = mask.copy()
    for i in range(1, num_lat - 1):
        for j in range(1, num_lon - 1):
            if mask[i, j] == 0:
                if np.any(mask[i - 1:i + 2, j - 1:j + 2] > 0):
                    mask_fixed[i, j] = mask[i - 1:i + 2, j - 1:j + 2].max()

    return mask_fixed
def mask_central_vs_marginal():
    """
    基于 纬度 + 多年平均 SIT 的 Arctic 分区
    0 = 非海洋
    1 = Central Thick Ice Region
    2 = Marginal Seas
    3 = 海洋
    """
    # ---------------------------
    # 1. 读取 PIOMAS 网格
    # ---------------------------
    file_path = r'E:\SeaIceDataSet\PIOMAS\codeTOnc\piomas_bin_reader-master\data\PIOMAS\PIOMAS_2011-2021_summer_7days_WithoutLand\avg_thickness_data_20110501_to_20110507.nc'
    ds = xr.open_dataset(file_path)

    lats = ds['latitude'].values
    lons = ds['longitude'].values.astype(float)

    # 转换经度至 -180~180
    lons[lons > 180] = lons[lons > 180] - 360

    num_lat, num_lon = lats.shape

    # ---------------------------
    # 2. 海陆掩膜
    # ---------------------------
    mask_land = np.zeros((num_lat, num_lon), dtype=int)
    for i in range(num_lat):
        for j in range(num_lon):
            if globe.is_ocean(lats[i, j], lons[i, j]):
                mask_land[i, j] = 3

    # 最终掩膜（0 = 非敏感区）
    mask = np.zeros((num_lat, num_lon), dtype=int)
    data = np.load("E:\code\Prediction_SIT\data\\before\\2011-2021_allyear.npz")
    sit = data["mod_da"]  # (4015, 120, 360)

    valid_ratio = np.sum(~np.isnan(sit), axis=0) / sit.shape[0]
    sit_mean = np.nanmean(
        np.where(mask_land[None, :, :] == 3, sit, np.nan),
        axis=0
    )
    # 有效性约束
    sit_mean[valid_ratio < 0.3] = np.nan


    # Central Thick Ice Region
    central = (
            (lats >= 80) &
            (sit_mean >= 2.0) &
            (mask_land == 3)
    )

    # Marginal Seas
    marginal = (
            ((lats < 80.0) | (sit_mean < 1)) &
            (mask_land == 3)
    )

    mask[central] = 1
    mask[marginal] = 2

    return mask

def compute_metrics(predictions, targets):
    targets = targets.permute(0, 1, 2, 3).detach().cpu().numpy()
    predictions = predictions.permute(0, 1, 2, 3).detach().cpu().numpy()

    batch_size = predictions.shape[0]
    Seq_len = predictions.shape[1]

    ssim = 0
    mask = mask_land()
    for batch in range(batch_size):
        for frame in range(Seq_len):
            target_frame = targets[batch, frame].squeeze()  # 当前目标帧
            pred_frame = predictions[batch, frame].squeeze()  # 当前预测帧
            target_frame = np.ma.masked_where(mask == 0, target_frame)  # 陆地区域变成 NaN
            pred_frame = np.ma.masked_where(mask == 0, pred_frame)
        ssim += structural_similarity(target_frame, pred_frame, data_range=100)

    ssim /= (batch_size * Seq_len)

    mse = MSE(predictions, targets,mask)
    mae = MAE(predictions, targets, mask)
    mae2 = MAE2(predictions, targets, mask)
    # 计算 R²
    r2 = compute_r2(predictions, targets, mask)

    return mse, ssim,mae,mae2,r2

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(args):

    cache_dir = os.path.join(args.res_dir, 'cache')
    check_dir(cache_dir)

    model_dir = os.path.join(args.res_dir, 'model')
    check_dir(model_dir)

    log_dir = os.path.join(args.res_dir, 'log')
    check_dir(log_dir)

    return cache_dir, model_dir, log_dir

def init_logger(log_dir):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(log_dir, time.strftime("%Y_%m_%d") + '.log'),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


