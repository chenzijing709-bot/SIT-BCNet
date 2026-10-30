import os
import shutil

#移动文件
def move_files(src_folder, dest_folder):
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            src_file_path = os.path.join(root, file)
            shutil.copy2(src_file_path, dest_folder)

src_folder = 'E:\SeaIceDataSet\CS2SMOS\data\\north'  # 替换为你的源文件夹路径
dest_folder = 'E:\SeaIceDataSet\PIOMAS\codeTOnc\piomas_bin_reader-master\CS2SMOSdata'  # 替换为你的目标文件夹路径
move_files(src_folder, dest_folder)

#修改文件的名称
folder_path = '/CS2SMOSdata'  # 替换为你文件夹的路径
for filename in os.listdir(folder_path):
    if filename.endswith('.nc'):
        date = filename.split('_')[5]
        new_filename = f'{date}.nc'
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))