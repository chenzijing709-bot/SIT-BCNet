import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import struct
import xarray as xr
from matplotlib import pyplot as plt

#加载经度('lon')和纬度('lat')的网格数据，并将它们存储在一个名为“grids”的字典中。


grids = {}
dims = (120, 360)
for i in ['lon', 'lat']:
    grid = np.array(pd.read_csv(f'../grids/{i}grid.dat',
                                header=None,
                                delim_whitespace=True))

    flat_grid = grid.ravel()

    shaped_grid = flat_grid.reshape(dims)

    grids[i] = shaped_grid

def day_to_date(year: int, day: int):
    # 创建新的日期对象，该日期是指定年份的1月1日
    date_obj = datetime(year, 1, 1)

    # 使用timedelta增加指定的天数
    # 需要减去1，因为1月1日是第一天
    date_obj = date_obj + timedelta(day - 1)

    # 使用strftime函数将日期对象转为格式化字符串
    date_str = date_obj.strftime('%m%d')

    return date_str
def process_piomas(year):
    binary_dir = f'E:\SeaIceDataSet\PIOMAS\codeTOnc\piomas_bin_reader-master\data\\new\\hiday.H{year}.gz'

    with open(binary_dir, mode='rb') as file:
        fileContent = file.read()
        data = struct.unpack("f" * (len(fileContent) // 4), fileContent)
        print(len(data))

        # Put it in a 3D array 365*120*360
        native_data = np.full((365, dims[0], dims[1]), np.nan, dtype=np)
        for day in range(1, 366):
            start = (day - 1) * (dims[0] * dims[1])
            end = day * (dims[0] * dims[1])
            thickness_list = np.array(data[start:end])

            gridded = thickness_list.reshape(dims[0], dims[1])


            native_data[day - 1, :, :] = gridded
            ds = xr.Dataset(data_vars={'thickness': (['x', 'y'], gridded)},

                            coords={'longitude': (['x', 'y'], grids['lon']),
                                    'latitude': (['x', 'y'], grids['lat'])})

            ds.attrs['data_name'] = 'Dayly mean Piomas sea ice thickness data'

            ds.attrs['description'] = """Sea ice thickness in meters on the native 360x120 grid"""

            output_dir = f'./new/'

            '''2010        1021 - 1231
               2011 - 2023 0101 - 0409 1015 - 1231'''

            '''if year==2010 :
                date_obj = datetime(2010, 10, 21)
                # 通过 timetuple().tm_yday 获取该日期是该年的第几天
                day_of_year = date_obj.timetuple().tm_yday
                if day>=day_of_year:
                    ds.to_netcdf(f'{output_dir}{year}{day_to_date(year, day)}.nc', 'w')
            else:
                date_obj = datetime(year, 4, 15)
                begin = date_obj.timetuple().tm_yday
                date_obj = datetime(year, 10, 15)
                end = date_obj.timetuple().tm_yday
                if day<= begin or day>=end:
                    ds.to_netcdf(f'{output_dir}{year}{day_to_date(year, day)}.nc', 'w')'''

            ds.to_netcdf(f'{output_dir}{year}{day_to_date(year, day)}.nc', 'w')

    return native_data


for year in range(2023, 2025):
    x = process_piomas(year)

