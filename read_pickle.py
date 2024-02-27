import pickle 
import os
import numpy as np
# with open('data/TJ4D/TJ4D_infos_train.pkl', 'rb') as f:
#     # 使用pickle.load()方法读取数据
#     data = pickle.load(f)
# print(1)
file_path = 'data/TJ4D/TJ4D_gt_database/271_Car_6.bin'
b = np.fromfile(file_path, dtype=np.float32)
print(b.shape)
b = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
print(1)