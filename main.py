# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。
import torch
from openprompt import PromptDataLoader, PromptForClassification

import pbl
import generate_goldlabel
import pandas as pd
from pbl import *
from fine_tuning import *

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # set training data and testing data
    # able to use few-shot method
    train_path = "./2012_Temporal_Relations_Challenge/2012-07-15.original-annotation.release"
    number_ON_label = 5
    number_OFF_label = 5
    window_size_before_target_sentence = 3
    window_size_after_target_sentence = 2
    train_dataset_prompt, train_dataset_tuning = generate_goldlabel.get_train_set(train_path, number_ON_label, number_OFF_label,
                                                                                  window_size_before_target_sentence,
                                                                                  window_size_after_target_sentence)

    test_path = "./2012_Temporal_Relations_Challenge/ground_truth/merged_i2b2"
    test_dataset_prompt, test_dataset_tuning = generate_goldlabel.get_test_set(test_path, window_size_before_target_sentence,
                                                                               window_size_after_target_sentence)


    plm, tokenizer, model_config, WrapperClass = pbl.select_plm(0)

    mytemplate, myverbalizer = pbl.select_template_and_verbalizer(0, tokenizer, plm)

    pbl_train_and_evaluate(train_dataset_prompt=train_dataset_prompt, validation_dataloader_prompt=test_dataset_prompt,
                               plm=plm, mytemplate=mytemplate, myverbalizer=myverbalizer, tokenizer=tokenizer,
                               WrapperClass=WrapperClass)

    fine_tuning_train_and_evaluate(train_dataset_tuning=train_dataset_tuning, test_dataset_tuning=test_dataset_tuning)