import sys
sys.path.append('../')
import torch
import argparse
import os
from utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, choices=['hdfs', 'bgl', 'tbird'])
parser.add_argument("-train_file", default=None)
parser.add_argument("-model_output", default=None)
parser.add_argument("-block_size", default=512, type=int)
parser.add_argument("-mlm_probability", default=0.15, type=float)
parser.add_argument("-learning_rate", default=5e-5, type=float)
parser.add_argument('-visible_gpus', default='-1', type=str)
parser.add_argument("-num_train_epochs", default=1, type=int)
parser.add_argument("-max_steps", default=-1, type=int)
parser.add_argument("-save_steps", default=500, type=int)
parser.add_argument("-train_batch_size", default=16, type=int)
parser.add_argument("-seed", default=21, type=int)
parser.add_argument("-resume_from_checkpoint", type=str2bool, nargs='?', const=True, default=False)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

os.environ["PYTHONHASHSEED"] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.model_output == None:
    if args.dataset == 'hdfs':
        args.model_output = '../../models/hdfs/span'
    elif args.dataset == 'bgl':
        args.model_output = '../../models/bgl/span'
    elif args.dataset == 'tbird':
        args.model_output = '../../models/tbird/span'

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers_modified import BertForMLMSBO

pretrained_model_str = 'bert-base-uncased'

bert_model = BertForMLMSBO.from_pretrained(pretrained_model_str)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_str)

# define the optimizer and learning rate schedule
optimizer = AdamW(bert_model.parameters(), lr=args.learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.max_steps)

from dataloader import LineByLineTextDataset, DataCollatorForLanguageModeling

print("Encoding train file using BERT tokenizer now, it could take a while if the training file is too large.")
train_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                      file_path=args.train_file,
                                      block_size=args.block_size)

data_collector = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    span=True,
    mlm_probability=args.mlm_probability)

from transformers_modified import TrainingArguments, Trainer

# Initiate trainer
trainArgs = TrainingArguments(
    output_dir=args.model_output,
    overwrite_output_dir=True,
    do_train=True,
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    learning_rate=args.learning_rate,
    save_strategy='steps',
    save_steps=args.save_steps,
    per_device_train_batch_size=args.train_batch_size,
    logging_steps=10
)

trainer = Trainer(
    model=bert_model,  # 模型对象
    args=trainArgs,  # 训练参数
    data_collator=data_collector,
    train_dataset=train_dataset,
    optimizers=(optimizer, scheduler),
)
# 预训练模型
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
trainer.save_model(os.path.join(args.model_output, 'LogDAPT_Span_' + args.dataset))
