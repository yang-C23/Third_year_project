import torch
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate, PrefixTuningTemplate
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer
from transformers import  AdamW, get_linear_schedule_with_warmup


from generate_goldlabel import get_train_set, get_test_set


# different pre-trained language model
def select_plm (option):
    if option == 0:
        return load_plm("bert", "bert-base-uncased")
    elif option == 1:
        return load_plm("roberta", "roberta-base")
    elif option == 2:
        return load_plm("t5", "t5-base")
    elif option == 3:
        return load_plm("gpt2", "gpt2")
    else :
        print("there is no option "+ option +"in PLM selection")
        return None

    #return format: plm, tokenizer, model_config, WrapperClass


#different template and verbalizer
#manual, soft, mixed, prefix
def select_template_and_verbalizer (option, tokenizer, plm):
    if option == 0:
        return (
        ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} Question: {"placeholder":"text_b"} were used between admission and discharge time. Is it correct? {"mask"}.'),
        ManualVerbalizer(tokenizer, num_classes=2, label_words=[["yes"], ["no"]])
        )
    elif option == 1:
        return (
        ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} In this paragraph of the note, {"placeholder":"text_b"} {"mask"} used between admission and discharge time.'),
        ManualVerbalizer(tokenizer, num_classes=2, label_words=[["is"], ["not"]])
        )
    elif option == 2:
        return (
        ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} In this paragraph of the note, {"placeholder":"text_b"} {"mask"} used between admission and discharge time.'),
        ManualVerbalizer(
        tokenizer = tokenizer,
        num_classes = 2,
        classes = ["ON", "OFF"],
        label_words = {
            "ON": ["definitely"],
            "OFF": ["possibly","probably","difinitely not"],
            })
        )
    elif option == 3:
        return (
        ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} Question: {"placeholder":"text_b"} were used between admission and discharge time. Is it correct? {"mask"}.'),
        SoftVerbalizer(tokenizer, plm, num_classes=2)
        )
    elif option == 4:
        return (
        #only for seq2seq, how about altoregressive model.
        PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} Question: {"placeholder":"text_b"} were used between admission and discharge time. Is it correct? {"mask"}'),
        SoftVerbalizer(tokenizer, plm, num_classes=2)
        )

    elif option == 5:
        return (
        MixedTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft": "Question:"} {"placeholder":"text_b"} {"soft"} {"soft"} {"soft"} {"soft"} {"soft"} Is it correct? {"mask"}.'),
        ManualVerbalizer(tokenizer, num_classes=2, label_words=[["yes"], ["no"]])
        )
    elif option == 6:
        return (
        MixedTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} In this paragraph of the note, {"placeholder":"text_b"} {"soft"} {"mask"} {"soft"} {"soft"} {"soft"} {"soft"} {"soft"} {"soft"}'),
        SoftVerbalizer(tokenizer, plm, num_classes=2)
        )
    else:
        print("we do not have option "+option)
        
        
def pbl_train_and_evaluate (train_dataset_prompt=None, validation_dataloader_prompt=None, plm=None, mytemplate=None, myverbalizer=None, tokenizer=None,
                            WrapperClass=None):
    train_dataloader_prompt = PromptDataLoader(dataset=train_dataset_prompt, template=mytemplate,
                                               tokenizer=tokenizer,
                                               tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                               decoder_max_length=3,
                                               batch_size=4, shuffle=True, teacher_forcing=False,
                                               predict_eos_token=False,
                                               truncate_method='head')

    # training
    use_cuda = True
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    if use_cuda:
        prompt_model = prompt_model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    for epoch in range(5):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader_prompt):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)

    # evaluation
    allpreds = []
    alllabels = []
    allids = []
    for step, inputs in enumerate(validation_dataloader_prompt):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        guids = inputs['guid']
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        allids.extend(guids.cpu().tolist())

    accuracy = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    errorIDs = []
    count = 0
    for i, j, id in zip(allpreds, alllabels, allids):
        count += 1
        if j == 1:
            if i == 1:
                TP += 1
            else:
                FN += 1
                errorIDs.append(id)
        else:
            if i == 1:
                FP += 1
                errorIDs.append(id)
            else:
                TN += 1

    print(TP, FN, FP, TN)
    # confusion matrix
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)

    print("overall accuracy:", accuracy)
    print("ON accuracy:     ", (TP) / (TP + FN))
    print("OFF accuracy:    ", (FP) / (FP + TN))
    print("Precision:       ", precision)
    print("Recall:          ", recall)
    print("F1:              ", F1)




if __name__ == '__main__':
    train_path = "./2012_Temporal_Relations_Challenge/2012-07-15.original-annotation.release"
    number_ON_label = 5
    number_OFF_label = 5
    window_size_before_target_sentence = 3
    window_size_after_target_sentence = 2
    train_dataset_prompt, train_dataset_tuning = get_train_set(train_path, number_ON_label, number_OFF_label,
                                                               window_size_before_target_sentence,
                                                               window_size_after_target_sentence)

    test_path = "./2012_Temporal_Relations_Challenge/ground_truth/merged_i2b2"
    test_dataset_prompt, test_dataset_tuning = get_test_set(test_path, window_size_before_target_sentence,
                                                            window_size_after_target_sentence)

    plm, tokenizer, model_config, WrapperClass = select_plm(2)
    mytemplate, myverbalizer = select_template_and_verbalizer(0, tokenizer, plm)

    pbl_train_and_evaluate(train_dataset_prompt = train_dataset_prompt, validation_dataloader_prompt = test_dataset_prompt, plm=plm , mytemplate = mytemplate, myverbalizer = myverbalizer, tokenizer = tokenizer, WrapperClass = WrapperClass)

