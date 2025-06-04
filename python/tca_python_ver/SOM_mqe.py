import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
import os
import pandas as pd

def counting_mqe(test_data,weight):
    
    test_data = scaler.transform(test_data)
    test_data_mqe = []
    
    # 計算MQE
    for i in range(test_data.shape[0]):  
        # 找到BMU（最佳匹配單元）
        bmu_x, bmu_y = som.winner(test_data[i,:])

        # 取得BMU的權重向量
        BMU_vector = weight[bmu_x, bmu_y]

        # 計算MQE（均方誤差）
        mqe = np.sqrt(np.sum((BMU_vector - test_data[i,:])**2))
        test_data_mqe.append(mqe)

    # 輸出MQE
    test_data_mqe = np.array(test_data_mqe)
    return test_data_mqe



if __name__=='__main__':
    # 指定要讀取的 CSV 檔案名稱
    file_name = "tca_source_normal_data.csv"  # 自行指定要讀取的檔案名稱，本實驗可不進行更名，注意資料夾位置正確就好
    folder_path = r"C:\Users\user\Documents\GitHub\test\python\tca_python_ver\gamma_1_0\Condition_6_ex" # 指定資料夾路徑

    # CSV 檔案的路徑
    file_path = os.path.join(folder_path, file_name)


    # 讀取 CSV
    df = pd.read_csv(file_path)
    original_indices = df.index.to_numpy()

    # 分割資料 50% 訓練 / 30% 驗證 / 20% 測試
    train_data, temp_data = train_test_split(df, test_size=0.5, random_state=42)  # 先拆出 50% 訓練
    val_data, test_data_normal = train_test_split(temp_data, test_size=0.4, random_state=42)  # 剩下的 50% 拆成 30% 驗證 & 20% 測試

    # 取得拆分後 val_data 的原始索引
    val_original_indices = val_data.index.to_numpy()

    test_data_normal = np.array(test_data_normal)

    scaler = StandardScaler()

    # 標準化數據
    train_data = scaler.fit_transform(train_data)
    # Step 1: 進行訓練
    som_train = train_data  # Transpose training data to match dimensions

    # 設定SOM參數
    dimensions = (20, 20)  # SOM網路
    num_neuron = dimensions[0] * dimensions[1]  # SOM神經元數量
    cover_steps = 2000  # 訓練步數
    init_neighbor = 3  # 初始鄰域大小
    topology_function = 'hexagonal'  # 層拓撲方式
    distance_function = 'euclidean'  # 距離計算方式

    # 初始化SOM
    som = MiniSom(x=dimensions[0], 
                  y=dimensions[1], 
                  input_len=train_data.shape[1], 
                  sigma=init_neighbor, 
                  learning_rate=0.5, 
                  topology=topology_function,
                  activation_distance=distance_function)

    # 訓練SOM
    som.train(som_train, cover_steps)

    # 提取權重
    weight = som.get_weights()  # 取得SOM的權重
    
    
    #vid data
    
    vid_data_mqe=counting_mqe(val_data,weight)
    
    print(vid_data_mqe.shape)

    
    
    #test data
    # 指定要讀取的 CSV 檔案名稱
    target_normal_file_name = "tca_target_normal_data.csv"  # 自行指定要讀取的檔案名稱
    target_abnormal_file_name = "tca_target_abnormal_data.csv"  # 自行指定要讀取的檔案名稱


    # 完整的 CSV 檔案路徑 target_normal
    target_normal_file_path = os.path.join(folder_path, target_normal_file_name)
    target_normal_test_df = pd.read_csv(target_normal_file_path)
    
    # 完整的 CSV 檔案路徑 target_abnormal
    target_abnormal_file_path = os.path.join(folder_path, target_abnormal_file_name)
    target_abnormal_test_df = pd.read_csv(target_abnormal_file_path)

    # 讀取異常測試資料
    test_data_target_normal=np.array(target_normal_test_df)
    test_data_target_abnormal = np.array(target_abnormal_test_df)


    #將百分之20的正常測試數據和百分之100的異常測試數據組合在一起，形成新的測試數據
    test_data = np.vstack([test_data_target_normal,test_data_target_abnormal])

    #test_data = test_data_target_abnormal

    test_data_mqe=counting_mqe(test_data,weight)
    
    print(test_data_mqe.shape)


    max_value = np.max(vid_data_mqe)
    max_index = np.argmax(vid_data_mqe)
    print("最大值:", max_value)
    print("在拆分後的 val_data 中位置:", max_index)

    # 使用 val_data 的 index 映射得到原始 CSV 的 sample 序號
    original_sample_index = val_original_indices[max_index]
    print("對應於原始 CSV 的 sample 序號:", original_sample_index)



    # MQE basline
    mean_mqe = np.mean(vid_data_mqe)
    std_mqe = np.std(vid_data_mqe)

    # 設定 baseline（通常是平均 + 3*標準差）
    mqe_baseline_3 = mean_mqe + 3 * std_mqe
    mqe_baseline_6 = mean_mqe + 6 * std_mqe

    # ========================
    # 準備資料
    mqe_normal = test_data_mqe[:len(test_data_target_normal)]      # 預期為「正常」的測試資料
    mqe_abnormal = test_data_mqe[len(test_data_target_normal):]    # 預期為「異常」的測試資料

    # ========================
    # 分類結果（0 = 正常, 1 = 異常）
    def classify_mqe(data_mqe, threshold):
        return (data_mqe > threshold).astype(int)

    # ========================
    # 使用 3 倍標準差
    pred_normal_3 = classify_mqe(mqe_normal, mqe_baseline_3)
    pred_abnormal_3 = classify_mqe(mqe_abnormal, mqe_baseline_3)

    # ========================
    # 使用 6 倍標準差
    pred_normal_6 = classify_mqe(mqe_normal, mqe_baseline_6)
    pred_abnormal_6 = classify_mqe(mqe_abnormal, mqe_baseline_6)

    # ========================
    # 計算分類正確率
    acc_normal_3 = np.mean(pred_normal_3 == 0)  # 正常被正確分類為正常
    acc_abnormal_3 = np.mean(pred_abnormal_3 == 1)  # 異常被正確分類為異常

    acc_normal_6 = np.mean(pred_normal_6 == 0)
    acc_abnormal_6 = np.mean(pred_abnormal_6 == 1)

    # ========================
    # 顯示結果
    print("=== 分類正確率（3倍標準差） ===")
    print(f"正常資料分類正確率: {acc_normal_3*100:.2f}%")
    print(f"異常資料分類正確率: {acc_abnormal_3*100:.2f}%")

    print("=== 分類正確率（6倍標準差） ===")
    print(f"正常資料分類正確率: {acc_normal_6*100:.2f}%")
    print(f"異常資料分類正確率: {acc_abnormal_6*100:.2f}%")

    import matplotlib.pyplot as plt
    
    # 設置中文支持
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用 Microsoft YaHei 字體
    plt.rcParams['axes.unicode_minus'] = False  # 顯示負號

    
    plt.figure(2)
    plt.plot(vid_data_mqe, color='g', marker='.', linestyle='', label='source_normal_data(225 normal)')
    plt.plot(test_data_mqe[0:750], color='b', marker='.', linestyle='', label='target_normal_data(750 normal)')
    plt.plot(test_data_mqe[750:], color='r', marker='.', linestyle='', label='target_abnormal_data(750 abnormal)')
    plt.axhline(y=mqe_baseline_3, color='m', linestyle='--', label=f'Baseline: 3倍標準差')
    plt.axhline(y=mqe_baseline_6, color='k', linestyle='--', label=f'Baseline: 6倍標準差')
    plt.xlabel('Samples')
    plt.ylabel('MQE')
    plt.legend()
    plt.title('馬達運轉軸承健康診斷')
    plt.grid()
    plt.show()
