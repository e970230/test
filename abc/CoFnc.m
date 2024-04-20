function costfunction=CoFnc(input_data,answer_data)
    costfunction=mse(answer_data-input_data);
end