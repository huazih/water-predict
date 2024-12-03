import csv

from sklearn.preprocessing import StandardScaler


from dataset.ExpotData import ExportData
from load_data import load_data

_, water_level_data = load_data()
seq_length = 7
myExport=ExportData(water_level_data,seq_length)


# 保存为 CSV 文件
with open('dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([f'day_{i+1}_water_position' for i in range(seq_length)] + ['label'])  # 写入列标题
    for i in range(len(myExport)):
        features, label = myExport[i]
        writer.writerow(features.tolist() + [label.item()])
