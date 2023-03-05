import sys
sys.path.append('../')
import torch
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
from matplotlib import pyplot as plt
import argparse
from utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, choices=['hdfs', 'bgl', 'tbird'])
parser.add_argument("-do_train", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-do_test", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-pretrained_model", default=None)  # '../../models/hdfs/mlm/checkpoint-5000'
parser.add_argument("-saved_model", default=None)  # '../../models/hdfs/mlm/checkpoint-5000/model.pth'
parser.add_argument("-train_file", default=None)  # '../../preprocessed_data/hdfs/train1000+1000.csv'
parser.add_argument("-test_file", default=None)  # '../../preprocessed_data/hdfs/test0.05.csv'
parser.add_argument("-train_batch_size", default=32, type=int)
parser.add_argument("-eval_batch_size", default=64, type=int)
parser.add_argument("-test_batch_size", default=32, type=int)
parser.add_argument("-learning_rate", default=5e-5, type=float)
parser.add_argument("-train_epochs", default=2, type=int)
parser.add_argument("-train_show_step", default=50, type=int)
parser.add_argument("-test_show_step", default=50, type=int)
parser.add_argument('-visible_gpu', default='0', type=str)
parser.add_argument("-seed", default=43, type=int)
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

os.environ["PYTHONHASHSEED"] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

pretrained_model_str = 'bert-base-uncased'

start_epoch = 1
stop_step = 10000  # 训练多少步结束 10000表示没有stop step

if args.pretrained_model is not None:
    save_model = os.path.join(args.pretrained_model, "model.pth")
else:
    save_model = "../../models/" + args.dataset + "/model.pth"

device = torch.device("cuda:" + args.visible_gpu) if torch.cuda.is_available() else torch.device("cpu")

if args.do_train:
    dataset = load_dataset("csv", data_files={'train': args.train_file, 'test': args.test_file, 'eval': args.test_file})
else:
    dataset = load_dataset("csv", data_files={'test': args.test_file})

if args.dataset == 'hdfs':
    dataset = dataset.remove_columns(["BlockId"])
dataset = dataset.rename_column("Label", "label")
dataset = dataset.rename_column("EventSequence", "text")
# print(type(dataset["train"][10]['text']))

from transformers import AutoModelForSequenceClassification

if not args.do_test:
    if args.pretrained_model is not None:
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_str, num_labels=2)
else:
    model = torch.load(args.saved_model)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_str)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
# print(tokenized_datasets["train"][1]['input_ids'])

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

from transformers import get_scheduler, optimization
# scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )# 如果有num_warmup_steps就先在预热阶段线性增加到optimizer里面设置的lr；若没有则直接从lr线性降低
# scheduler = optimization.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
# scheduler = optimization.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps, num_cycles=1)


def get_acc_rec_f1(labels, predictions):
    accuracy = accuracy_score(labels, predictions)  # 默认1为阳性，而数据集中1表示异常日志序列
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='binary', zero_division=0)  # 不能写macro
    return round(accuracy, 4), round(precision, 4), round(recall, 4), round(macro_f1, 4)


def test():
    small_test_dataset = tokenized_datasets["test"].shuffle(seed=args.seed)
    test_dataloader = DataLoader(small_test_dataset, drop_last=True, batch_size=args.test_batch_size)
    num_testing_steps = len(test_dataloader)
    model.eval()
    total_test_step = 1
    total_preds = []
    total_labels = []
    print("start testing...")
    progress_bar = tqdm(range(num_testing_steps))
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predictions = predictions.to("cpu").tolist()
        labels = batch["labels"].to("cpu").tolist()
        total_preds.extend(predictions)
        total_labels.extend(labels)

        progress_bar.update(1)

        if total_test_step % args.test_show_step == 0:
            accuracy_tillnow, precision_tillnow, recall_tillnow, macro_f1_tillnow = get_acc_rec_f1(total_labels, total_preds)
            print("Until step {}, accuracy is {}, precision is {}, recall is {}, F1-Score is {}".format(total_test_step, accuracy_tillnow,
                                                                                                        precision_tillnow, recall_tillnow,
                                                                                                        macro_f1_tillnow))
        total_test_step += 1
    progress_bar.close()
    print("total labels length {}".format(len(total_labels)))
    print("total predictions length {}".format(len(total_preds)))
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for k in range(len(total_labels)):
        if total_labels[k] == 1 and total_preds[k] == 1:
            tp += 1
        elif total_labels[k] == 1 and total_preds[k] == 0:
            fn += 1
        elif total_labels[k] == 0 and total_preds[k] == 1:
            fp += 1
        elif total_labels[k] == 0 and total_preds[k] == 0:
            tn += 1
    print("tp={}, fn={}, fp={}, tn={}".format(tp, fn, fp, tn))
    # calc_accuracy = (tp+tn)/(tp+fn+fp+tn)
    # calc_precision = tp/(tp+fp)
    # calc_recall = tp/(tp+fn)
    # calc_macro_f1 = 2/((1 / calc_precision) + (1 / calc_recall))
    # print("My accuracy is {}, precision is {}, recall is {}, F1-Score is {}".format(calc_accuracy, calc_precision, calc_recall, calc_macro_f1))

    total_accuracy, total_precision, total_recall, total_macro_f1 = get_acc_rec_f1(total_labels, total_preds)
    print("In the testing set, accuracy is {}, precision is {}, recall is {}, F1-Score is {}".format(total_accuracy, total_precision,
                                                                                                     total_recall, total_macro_f1))


def train():
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=args.seed)
    small_eval_dataset = tokenized_datasets["eval"].shuffle(seed=args.seed)
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, drop_last=True, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, drop_last=True, batch_size=args.eval_batch_size)
    num_training_steps = stop_step if not stop_step == 10000 else args.train_epochs * len(train_dataloader)
    criterion = torch.nn.CrossEntropyLoss()
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    total_preds = []
    total_labels = []
    total_preds_eval = []
    total_labels_eval = []
    x = []
    y = []
    y_eval = []
    total_train_step = 1
    for epoch in range(start_epoch, start_epoch + args.train_epochs):
        for batch, batch_eval in zip(train_dataloader, eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = criterion(outputs.logits, batch["labels"])  # lr=5e-4时loss无法减小
            loss.backward()

            x.append(total_train_step)
            y.append(loss.item())

            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions.to("cpu").tolist()
            labels = batch["labels"].to("cpu").tolist()
            total_preds.extend(predictions)
            total_labels.extend(labels)

            batch_eval = {k: v.to(device) for k, v in batch_eval.items()}
            with torch.no_grad():
                outputs_eval = model(**batch_eval)
            loss_eval = criterion(outputs_eval.logits, batch_eval["labels"])
            y_eval.append(loss_eval.item())

            if total_train_step % args.train_show_step == 0:
                # plt.plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=4)
                # plt.plot(x, y_eval, color='orange', marker='o', linestyle='dashed', linewidth=1, markersize=4)
                # plt.show()
                accuracy_tillnow, prediction_tillnow, recall_tillnow, macro_f1_tillnow = get_acc_rec_f1(total_labels, total_preds)
                print("Until step {}, accuracy is {}, precision is {}, recall is {}, F1-Score is {}".format(total_train_step,
                                                                                                            accuracy_tillnow,
                                                                                                            prediction_tillnow,
                                                                                                            recall_tillnow,
                                                                                                            macro_f1_tillnow))

            progress_bar.update(1)
            total_train_step += 1
        if epoch >= args.train_epochs:
            torch.save(model, save_model)
            break
    # plt.plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=4)
    # plt.plot(x, y_eval, color='orange', marker='o', linestyle='dashed', linewidth=1, markersize=4)
    # plt.show()
    progress_bar.close()
    total_accuracy, total_precision, total_recall, total_macro_f1 = get_acc_rec_f1(total_labels, total_preds)
    print("In the training set, accuracy is {}, precision is {}, recall is {}, F1-Score is {}".format(total_accuracy, total_precision,
                                                                                                      total_recall, total_macro_f1))


if args.do_train:
    train()
    test()

if args.do_test:
    test()
