clc
clear
close all

% 定義環境參數
num_actions = 3;  % 替換為你的動作數量
state_dim = 5;    % 替換為你的狀態維度

% 建立演員模型
actor = [
    featureInputLayer(state_dim, 'Normalization', 'none', 'Name', 'actor_state')
    fullyConnectedLayer(128, 'Name', 'actor_fc1', 'WeightsInitializer', 'glorot', 'BiasInitializer', 'zeros')
    reluLayer('Name', 'actor_relu1')
    fullyConnectedLayer(64, 'Name', 'actor_fc2', 'WeightsInitializer', 'glorot', 'BiasInitializer', 'zeros')
    reluLayer('Name', 'actor_relu2')
    fullyConnectedLayer(num_actions, 'Name', 'actor_output', 'WeightsInitializer', 'glorot', 'BiasInitializer', 'zeros')
    softmaxLayer('Name', 'softmax')
];

% 建立評論家模型
critic = [
    featureInputLayer(state_dim, 'Normalization', 'none', 'Name', 'critic_state')
    fullyConnectedLayer(128, 'Name', 'critic_fc1', 'WeightsInitializer', 'glorot', 'BiasInitializer', 'zeros')
    reluLayer('Name', 'critic_relu1')
    fullyConnectedLayer(64, 'Name', 'critic_fc2', 'WeightsInitializer', 'glorot', 'BiasInitializer', 'zeros')
    reluLayer('Name', 'critic_relu2')
    fullyConnectedLayer(1, 'Name', 'critic_output', 'WeightsInitializer', 'glorot', 'BiasInitializer', 'zeros')
];

% 連接演員和評論家模型
actor_input = 'input_state';
critic_input = 'critic_state';

actor_critic = layerGraph(actor);
%%
actor_critic = addLayers(actor_critic, critic);

% 定義連接
% actor_critic = connectLayers(actor_critic, 'softmax', 'critic_fc1');
% actor_critic = connectLayers(actor_critic, 'actor_state', 'critic_state');

% 編譯模型
options = rlRepresentationOptions('LearnRate', 0.01);
actor_critic = rlRepresentation(actor_critic, options);

% 打印模型摘要
disp(actor_critic)
