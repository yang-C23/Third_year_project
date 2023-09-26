import os
import re
import linecache
import random
from itertools import zip_longest
from random import shuffle
from openprompt.data_utils import InputExample

def gold_labeling(path, window_size_before_target_sentence, window_size_after_target_sentence):
    files = os.listdir(path)
    files.sort()

    dataset = {}
    treatment_list = []
    tlink_list = []
    admission = ""
    discharge = ""
    for file in files:

        # extract treatment entities and time information
        if file.endswith(".extent"):

            treatment_list = []

            f = open(path + "/" + file)
            for line in f:
                s = re.split(r'[||""\'\']', line)
                if s[0] == "EVENT=":
                    if s[5] == "TREATMENT":
                        treatment_list.append(s[1])
                elif s[0] == "SECTIME=":
                    if s[5] == "ADMISSION":
                        admission = s[1]
                    elif s[5] == "DISCHARGE":
                        discharge = s[1]

        # extract temporal link information
        elif file.endswith(".tlink"):
            tlink_list = []

            t = open(path + "/" + file)
            for line in t:
                tlink = re.split(r'[||""\'\']', line)
                for i in treatment_list:
                    if tlink[1] == i:
                        begin = [int(x) for x in tlink[2].split()[0].split(":")]
                        end = [int(y) for y in tlink[2].split()[1].split(":")]
                        txt_path = (path + "/" + file).replace(".tlink", ".txt")
                        line_number = begin[0] - 1
                        txt_content = [x.replace("\n", " ") for x in linecache.getlines(txt_path)]
                        text_a = txt_content[:4] + [". Dorctor notes: "]

                        # context window
                        # One sentence before: One sentence after
                        # [line_number-1 : line_number+2]
                        target_sentent = txt_content[line_number - window_size_before_target_sentence : (line_number + 1) + window_size_after_target_sentence]
                        text_a += target_sentent
                        if tlink[5] == admission and tlink[9] == "BEFORE":
                            tlink_list.append([i, "OFF", begin, end, "".join(text_a)])
                        elif tlink[5] == admission and tlink[9] == "AFTER":
                            tlink_list.append([i, "ON", begin, end, "".join(text_a)])
                        elif tlink[5] == discharge and tlink[9] == "BEFORE":
                            tlink_list.append([i, "ON", begin, end, "".join(text_a)])
                        else:
                            pass

            dataset[file] = tlink_list

    return dataset


def get_train_set(train_path, number_ON_label, number_OFF_label, window_size_before_target_sentence,
                  window_size_after_target_sentence):
    # train_path = "/content/drive/MyDrive/3rd_year_project/2012_Temporal_Relations_Challenge/2012-07-15.original-annotation.release"

    train_data = gold_labeling(train_path, window_size_before_target_sentence, window_size_after_target_sentence)
    # print(train_data)
    label_ON_dataset = []
    label_OFF_dataset = []

    train_dataset_prompt = []
    train_dataset_fine = []
    count = 0
    train_ON_count = 0
    train_OFF_count = 0
    total_number_ON = 0
    total_number_OFF = 0
    for file in train_data:

        for gold_label in train_data[file]:
            if gold_label[1] == "ON":
                label_ON_dataset.append([file] + gold_label)
                total_number_ON += 1
            else:
                label_OFF_dataset.append([file] + gold_label)
                total_number_OFF += 1

    # few-shot training set
    if number_ON_label <= total_number_ON:
        train_label_ON_dataset = random.sample(label_ON_dataset, number_ON_label)
    else:
        train_label_ON_dataset = label_ON_dataset
    if number_OFF_label <= total_number_OFF:
        train_label_OFF_dataset = random.sample(label_OFF_dataset, number_OFF_label)
    else:
        train_label_OFF_dataset = label_OFF_dataset

    # randomly merge label ON and OFF dataset
    print("train_label_ON_dataset", len(train_label_ON_dataset))
    print("train_label_OFF_dataset", len(train_label_OFF_dataset))
    temp_list = list(zip_longest(train_label_ON_dataset, train_label_OFF_dataset, fillvalue=None))
    shuffle(temp_list)

    merged_label_dataset = []
    for item in temp_list:
        if item[0] is not None:
            merged_label_dataset.append(item[0])
        if item[1] is not None:
            merged_label_dataset.append(item[1])

    print(len(merged_label_dataset))
    for gold_label in merged_label_dataset:
        if gold_label[2] == "ON":
            train_label = 1
            train_ON_count += 1
        else:
            train_label = 0
            train_OFF_count += 1

        # generate training data for prompt-based leanring
        input_example = InputExample(text_a=gold_label[5], text_b=gold_label[1], label=train_label, guid=count,
                                     meta=gold_label[0])
        count += 1
        # generate training data for fine-tuning
        train_dataset_prompt.append(input_example)
        train_dataset_fine.append([gold_label[5], gold_label[1], train_label, count, gold_label[0]])

    print(input_example)
    print("number of ON train", train_ON_count)
    print("number of OFF train", train_OFF_count)
    print("total number of training set", count)

    return (train_dataset_prompt, train_dataset_fine)


def get_test_set(test_path, window_size_before_target_sentence, window_size_after_target_sentence):
    test_ON_count = 0
    test_OFF_count = 0
    # test_path = "/content/drive/MyDrive/3rd_year_project/2012_Temporal_Relations_Challenge/ground_truth/merged_i2b2"
    test_data = gold_labeling(test_path, window_size_before_target_sentence, window_size_after_target_sentence)
    # print(test_data)
    test_dataset_prompt = []
    test_dataset_fine = []
    count = 0
    for file in test_data:

        # creat training labels
        for gold_label in test_data[file]:
            if gold_label[1] == "ON":
                test_label = 1
                test_ON_count += 1
            else:
                test_label = 0
                test_OFF_count += 1

            input_example = InputExample(text_a=gold_label[4], text_b=gold_label[0], label=test_label, guid=count,
                                         meta=file)
            count += 1
            test_dataset_prompt.append(input_example)
            test_dataset_fine.append([gold_label[4], gold_label[0], test_label, count, file])

    print("number of ON test", test_ON_count)

    print("number of OFF test", test_OFF_count)

    return (test_dataset_prompt, test_dataset_fine)

# if __name__ == '__main__':
#
