import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
import scipy.io

# 此為ExtraTreesRegressor的範例程式，下方需決定極限樹的模型設定和交叉驗證的fold數等才能使用。

# 在使用時須先注意若要運行此程式需先import下列模組才能使用，當下測試模組版本和python版本會記錄下來做為參考
#    測試運行當下python版本為 3.11.2
#    1. numpy(1.26.4)
#    2. sklearn(1.5.0)
#    3. scipy(1.14.0)


# 最後修改時間:2024/8/28 


#------------------------------------------示範程式----------------------------------------
# 出處: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
'''
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
reg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(
   X_train, y_train)
reg.score(X_test, y_test)

'''
#-------------------------------------------讀取檔案部分---------------------------------------


'''
#讀取原本預壓力檔案，並未進行篩選特徵之完整檔案
mat = scipy.io.loadmat('feature_dataset_heavy.mat')
feature_dataset = mat['feature_dataset']            #此原數據之輸入要求為樣本*特徵

'''
#透過LTFRM選取最佳前30特徵的預壓力資料

mat = scipy.io.loadmat('feature_dataset_top30.mat')

feature_dataset = mat['Data']

init_data = feature_dataset[:, 1:]      # 擷取原數據的特徵，第0列為標籤所以特徵從第1列開始擷取
label = feature_dataset[:, 0]           # 擷取原數據的標籤，為原數據的第0列


#-------------------------------------------主要運行程式區域---------------------------------------


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)     #設定sKfold交叉驗證模組(拆分的組數，再拆分前是否打亂，隨機性設定)

unique_numbers = np.unique(label)       #將標籤中不相同處給區別出來，以後續處理使用
final_vr_answer = np.empty((0, len(unique_numbers)))


test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
all_mse = []
for train_index_test_ver, test_index_test_ver in test_skf.split(init_data,label):
        X_train, X_test = init_data[train_index_test_ver], init_data[test_index_test_ver]       #將數據拆分成訓練數據和測試數據，並透過Kfold交叉驗證方式進行區分
        y_train, y_test = label[train_index_test_ver], label[test_index_test_ver]               #將數據拆分成訓練標籤和測試標籤，並透過Kfold交叉驗證方式進行區分
        
        test_model = ExtraTreesRegressor(n_estimators=100,                   #樹的數量
                                         max_features=3,                     #分裂時考慮的最大特徵數
                                         min_samples_split=2,                #葉節點最小樣本數
                                         random_state=42)
        
        test_model.fit(X_train, y_train)        #訓練模型
        y_pred = test_model.predict(X_test)     #預測解答


        # 計算每種標籤預測答案和原始標籤之MSE值
        verify_mse = {er_label: mean_squared_error(y_test[y_test == er_label], y_pred[y_test == er_label]) for er_label in unique_numbers}
        temporary_vr_answer = []  # 初始化為一個空列表，存儲每次的vr_answer

        #將每種標籤各自的MSE值進行判斷
        for er_label, verify_mse in verify_mse.items():
            if verify_mse >= er_label:  #若MSE過大則以一個大數去代替
                vr_answer = 1000
            else:
                vr_answer = verify_mse / er_label * 100 #若MSE正常則計算其錯誤百分比

            temporary_vr_answer.append(vr_answer)  # 使用列表的append方法將vr_answer追加到final_vr_answer中
        

        temporary_vr_answer = np.array(temporary_vr_answer).reshape(1, -1)  # 將其轉換為一行
        final_vr_answer = np.vstack((final_vr_answer, temporary_vr_answer)) # 垂直堆疊

        # 計算預測答案和原始標籤之MSE值
        mse = mean_squared_error(y_test, y_pred)    #計算MSE值
        all_mse.append(mse)                         #將當次MSE值記錄下來
    


row_means = np.mean(final_vr_answer, axis=0)    #取每個fold中同種標籤種類的MSE
final_mse_mean = np.mean(all_mse, axis=0)       #將所有記錄下來的MSE值進行平均

# 輸出每個標籤的mse值
# 假設 unique_numbers 是你的標籤對應的數據
for er_label, row_mean in zip(unique_numbers, row_means):
    if row_mean >= er_label:
        er_answer = "誤差率過大"
    else:
        er_answer = f"誤差率百分之 {row_mean / er_label * 100:.2f}"

    print(f"預壓力 {er_label} 的MSE值為 {row_mean:.2f} {er_answer}")

print("預測模型驗證MSE值:", final_mse_mean)

