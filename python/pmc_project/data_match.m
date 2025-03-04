clc
clear
close all
%%
% 
% a = load("F10000_Labels_name.mat").Labels_name;
% b = load("F16000_Labels_name.mat").Labels_name;
% 
% Final_Labels_name = [a b];


%%  
F_total = [10000 16000];
model_total = 1:3;


for it_one = 1:size(model_total,2)

    model = model_total(it_one);
    
   

    folder = ['C:\Users\user\Desktop\desktop_pmc\訊號資料\nor_data_70_30\訓練資料'];  % 資料檔案所在的資料夾
    filePattern = fullfile(folder, ['input_data_models_' num2str(model) '_F_*.mat']);      % 定義匹配的文件類型 (*.mat)
    matFiles = dir(filePattern);                  % 列出所有符合的文件
    
    
    
    for k = 1:length(matFiles)
        baseFileName = matFiles(k).name;          % 取得每個檔案名稱
        fullFileName = fullfile(folder, baseFileName); % 完整檔案路徑
        data{k} = load(fullFileName);                % 加載檔案資料
        % 這裡可以對資料進行操作
    end
    
    Final_input_data = [data{1}.train_input_data;data{2}.train_input_data];

    save_input_file=['Final_train_nor_input_data_models_' num2str(model) '.mat'];  %存檔檔名
    save(fullfile(save_input_file),'Final_input_data');
end


%%
F_total = [10000 16000];
model_total = 1:3;


for it_one = 1:size(model_total,2)

    model = model_total(it_one);
    
   

    folder = ['C:\Users\user\Desktop\desktop_pmc\訊號資料\nor_data_70_30\訓練資料'];  % 資料檔案所在的資料夾
    filePattern = fullfile(folder, ['output_data_models_' num2str(model) '_F_*.mat']);      % 定義匹配的文件類型 (*.mat)
    matFiles = dir(filePattern);                  % 列出所有符合的文件
    
    
    
    for k = 1:length(matFiles)
        baseFileName = matFiles(k).name;          % 取得每個檔案名稱
        fullFileName = fullfile(folder, baseFileName); % 完整檔案路徑
        data{k} = load(fullFileName);                % 加載檔案資料
        % 這裡可以對資料進行操作
    end
    
    Final_output_data = mean([data{1}.train_output_data;data{2}.train_output_data],2);

    save_input_file=['Final_train_nor_output_data_models_' num2str(model) '.mat'];  %存檔檔名
    save(fullfile(save_input_file),'Final_output_data');
end

%%
F_total = [10000 16000];
model_total = 1:3;


for it_one = 1:size(model_total,2)

    model = model_total(it_one);
    
   

    folder = ['C:\Users\user\Desktop\desktop_pmc\訊號資料\nor_data_70_30\訓練資料'];  % 資料檔案所在的資料夾
    filePattern = fullfile(folder, ['test_input_models_' num2str(model) '_F_*.mat']);      % 定義匹配的文件類型 (*.mat)
    matFiles = dir(filePattern);                  % 列出所有符合的文件
    
    
    
    for k = 1:length(matFiles)
        baseFileName = matFiles(k).name;          % 取得每個檔案名稱
        fullFileName = fullfile(folder, baseFileName); % 完整檔案路徑
        data{k} = load(fullFileName);                % 加載檔案資料
        % 這裡可以對資料進行操作
    end
    
    Final_input_data = [data{1}.test_input_data;data{2}.test_input_data];

    save_input_file=['Final_test_nor_input_data_models_' num2str(model) '.mat'];  %存檔檔名
    save(fullfile(save_input_file),'Final_input_data');
end



%%

clc
clear
close all



F_total = [10000 16000];
model_total = 1:3;


for it_one = 1:size(model_total,2)

    model = model_total(it_one);
    
   

    folder = ['C:\Users\user\Desktop\desktop_pmc\訊號資料\nor_data_70_30\訓練資料'];  % 資料檔案所在的資料夾
    filePattern = fullfile(folder, ['test_output_models_' num2str(model) '_F_*.mat']);      % 定義匹配的文件類型 (*.mat)
    matFiles = dir(filePattern);                  % 列出所有符合的文件
    
    
    
    for k = 1:length(matFiles)
        baseFileName = matFiles(k).name;          % 取得每個檔案名稱
        fullFileName = fullfile(folder, baseFileName); % 完整檔案路徑
        data{k} = load(fullFileName);                % 加載檔案資料
        % 這裡可以對資料進行操作
    end
    
    Final_output_data = mean([data{1}.test_output_data;data{2}.test_output_data],2);

    save_input_file=['Final_test_nor_output_data_models_' num2str(model) '.mat'];  %存檔檔名
    save(fullfile(save_input_file),'Final_output_data');
end
