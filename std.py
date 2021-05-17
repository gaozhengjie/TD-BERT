# coding:utf-8

import numpy as np

with open('result.txt', 'r', encoding='utf-8') as f:
	data = f.readlines()
	data = list(map(lambda x: x.strip(), data))
	std = np.std(np.array(data).astype(float), ddof=1)
	std_2 = np.std(np.array(data).astype(float), ddof=0)  # 除以 n
	max_mean = np.mean(np.array(data).astype(float))	
	print('总共：{} 个结果, 均值：{:.4f}, 样本标准差： {:.4f}, 总体标准差: {:.4f}'.format(len(data), max_mean, std, std_2))
