import numpy as np
import scipy.io

def generate_dynamic_gene_space(num_params):
    predefined_ranges = []
    for _ in range(num_params):
        low = 1
        high = Dimensions[1]
        predefined_ranges.append({'low': low, 'high': high})
    return predefined_ranges


mat = scipy.io.loadmat('feature_dataset_heavy.mat')

feature_dataset = mat['feature_dataset']

test = np.array([1,2,3,5,7,10])

data = feature_dataset[:, test]  # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]  # 擷取原數據的標籤，為原數據的第0列

Dimensions = np.shape(data)

    

# 示例：生成 7 个超参数上下限
num_params = 30

gene_space = [
    {'low': 10, 'high': 300},  # n_estimators
    {'low': 1, 'high': 30},    # max_features
    {'low': 2, 'high': 20}     # min_samples_split
]

gene_space_next = generate_dynamic_gene_space(num_params)

final_gene_space = gene_space + gene_space_next

print(final_gene_space)
