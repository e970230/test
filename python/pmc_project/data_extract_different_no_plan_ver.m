%pmc主軸不平衡數據實驗
%前半段會先進行(資料整合、特徵篩選、特徵抽取)三步得出初始的data後
%再行拆分將數據百分之50為訓練數據，百分之50為測試數據


clc
clear
close all

%% 資料整理

%資料輸入的部分只分成三種資料分別是(振動訊號、電流訊號、位移訊號)
%再切分數據時已將模型1、模型2、模型3和F10000和F16000都單獨區分開來了
%故這裡輸入的三種資料只需是同種模型的同進給數據即可進行另一種實驗

rng(0)
F_total = [10000 16000];
model_total = 1:3;

for it_one = 1:size(model_total,2)

    model = model_total(it_one);
    for it_two = 1:size(F_total,2)

        F = F_total(it_two);
        train_output_data = [];
        test_output_data = [];
        train_input_data = [];
        test_input_data = [];
        
        
        folder = ['C:\Users\user\Desktop\desktop_pmc\訊號資料\nor_data_70_30\原始資料'];  % 資料檔案所在的資料夾
        filePattern = fullfile(folder, ['*_data_models_' num2str(model) '_dlp_*_F_' num2str(F) '_nor.mat']);      % 定義匹配的文件類型 (*.mat)
        matFiles = dir(filePattern);                  % 列出所有符合的文件
        
        
        for k = 1:length(matFiles)
            baseFileName = matFiles(k).name;          % 取得每個檔案名稱
            fullFileName = fullfile(folder, baseFileName); % 完整檔案路徑
            data{k} = load(fullFileName);                % 加載檔案資料
            % 這裡可以對資料進行操作
        end
        
        
        
        %----------------------位移訊號資料--------------------------
        %這裡大小打500/2是要各分一半的資料，一半訓練一半測試
        size_data_num = fix(496*0.7);           %訓練數據比例
        size_data_num_test = 496-size_data_num;
        random_numbers = randperm(496);         %將樣本的編號順序隨機打亂
        for od = 1:6
            od_data = data{od+6}.Fianl_labels;
            train_output_data((size_data_num*(od-1)+1):od*size_data_num,1) = od_data(random_numbers(1:size_data_num),:);
            test_output_data((size_data_num_test*(od-1)+1):od*size_data_num_test,1) = od_data(random_numbers(size_data_num+1:end),:);

        end
        
        
        
        %------------------------------------------------------
        for id = 1:6
            id_data = data{id}.Final_input_data;
            train_input_data((size_data_num*(id-1)+1):id*size_data_num,:) = id_data(random_numbers(1:size_data_num),:);
            test_input_data((size_data_num_test*(id-1)+1):id*size_data_num_test,:) = id_data(random_numbers(size_data_num+1:end),:);
        end
        
        %----------------------------------------------------------

        
        save_input_file=['input_data_models_' num2str(model) '_F_' num2str(F) '.mat'];  %存檔檔名
        
        save_output_file=['output_data_models_' num2str(model) '_F_' num2str(F) '.mat'];  %存檔檔名
        
        save_test_input = ['test_input_models_' num2str(model) '_F_' num2str(F) '.mat'];

        save_test_output = ['test_output_models_' num2str(model) '_F_' num2str(F) '.mat'];
        
        save(fullfile(save_input_file),'train_input_data');
        
        save(fullfile(save_output_file),'train_output_data');

        save(fullfile(save_test_input),'test_input_data');

        save(fullfile(save_test_output),'test_output_data');
    end
end