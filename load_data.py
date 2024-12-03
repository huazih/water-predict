
from netCDF4 import Dataset as ncDataset
import numpy as np
from datetime import datetime,timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split, DataLoader

from dataset.WaterLevelDataset import WaterLevelDataset


def load_data_waterF(nc_file):
    nc_data = ncDataset(nc_file)
    variables_names = [
        "evpa",
        "fgrnd",
        "fros",
        "gstroage",
        "prec",
        "qinfl",
        "qlat",
        "qsmelt",
        "rain",
        "rgrd",
        "rsfc",
        "rtot",
        "sena",
        "seng",
        "senl",
        "smf_ice",
        "smf_soisno",
        "snow",
        "snowd",
        "snowevp",
        "soilt",
        "soilw",
        "subl",
        "swe"
    ]
    variables = [nc_data[var][:] for var in variables_names]  # 示例变量
    for i, var in enumerate(variables):
        if var.ndim==3:
            variables[i]=np.expand_dims(var,axis=1)
        # print(f"Variable shape {i}: {variables[i].shape}")
    combined_data = np.concatenate(variables, axis=1)
    return combined_data
def load_data_climateD(nc_file):
    nc_data = ncDataset(nc_file)
    variables_names = [
        "lwr",
        "swr",
        "sp",
        "prec",
        "rh2m",
        "q2m",
        "t2m",
        "u10",
        "v10"
    ]
    variables = [nc_data[var][:] for var in variables_names]
    # for i, var in enumerate(variables):
    #     print(f"Variable shape {i}: {variables[i].shape}")
    variables=np.expand_dims(variables,axis=0)
    combined_data = np.concatenate(variables, axis=0)
    return combined_data
def load_data():
    start_date=datetime(2021,6,1)
    end_date=datetime(2021,9,30)
    cur=start_date
    pre_waterf="water_feature/2020_2021/"
    pre_climated="climate_data/2021/"
    combine_data=[]
    while cur<=end_date:
        cur_str=cur.strftime("%Y%m%d")
        waterf_p=pre_waterf+cur_str+"_resulttest.nc"
        waterf_d=load_data_waterF(waterf_p)
        climated_p=pre_climated+cur_str+".nc"
        climated_d=load_data_climateD(climated_p)
        climated_d=np.expand_dims(climated_d,axis=0)
        cur_data=np.concatenate([waterf_d,climated_d],axis=1)
        combine_data.append(cur_data)
        cur=cur+timedelta(days=1)
    combine_data=np.concatenate(combine_data,axis=0)
    target_data=load_data_waterP()
    return combine_data,target_data
def load_data_waterP():
    # 读取 Excel 文件
    file_path = "water_positon/data/2021.xls"  # 替换为你的 Excel 文件路径
    df = pd.read_excel(file_path)  # 默认读取第一个工作表
    data=[]
    for column in df.columns[6:10]:
        column_data = df[column].tolist()  # 将列数据转换为列表
        column_data=[v for v in column_data if pd.notna(v)]
        data.extend(column_data)
    data=np.array(data)
    return data

# combined_data, water_level_data = load_data()
# scaler = StandardScaler()
# combined_data = scaler.fit_transform(combined_data)
# seq_length = 7
# dataset = WaterLevelDataset(combined_data, water_level_data, seq_length)
#
# # 划分训练集和验证集
# train_size = int(0.8 * len(dataset))  # 80% 训练集
# val_size = len(dataset) - train_size   # 20% 验证集
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# for inputs, labels in train_loader:
#     print(inputs.shape)
#     print(labels.shape)

