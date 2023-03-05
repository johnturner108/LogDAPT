import sys

sys.path.append('../')

import argparse
import os
import re
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument("-input_dir", default='../../datasets/hdfs/')
parser.add_argument("-output_dir", default='../../preprocessed_data/hdfs/')
parser.add_argument("-log_file", default='HDFS.log')
parser.add_argument("-train_ratio", default=0.8, type=float)
parser.add_argument("-test_small_ratio", default=0.05, type=float)
parser.add_argument("-shuffle", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-small_train_len", default=2000, type=int)
parser.add_argument("-small_train_len_normal", default=1000, type=int)
parser.add_argument("-dapt_train_steps", default=5000, type=int)
parser.add_argument("-dapt_batch_size", default=16, type=int)
parser.add_argument("-eval_size", default=5000, type=int)
parser.add_argument("-seed", default=21, type=int)

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir  # The output directory of preprocessing results
log_file = args.log_file  # The input log file name

input_file = os.path.join(input_dir, log_file)
log_structured_file = os.path.join(output_dir, log_file + "_structured.csv")
log_sequence_file = os.path.join(output_dir, log_file + "_sequence.csv")
# 在log_sequence_file基础上获得BlockId对应的标签
log_sequence_label_file = os.path.join(output_dir, log_file + "_sequence_label.csv")


def preprocess(input_file, log_structured_file, log_format):
    def generate_logformat_regex(logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def log_to_dataframe(log_file, regex, headers):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        cnt = 0
        with open(log_file, 'r') as fin:
            lines = fin.readlines()
            for line in tqdm(lines, total=len(lines)):
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    print("\n", line)
                    print(e)
                    pass
        print("Total size of logs is", linecount, 'out of', cnt, end='.\n')
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    print("Converting log to dataframe...")
    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(input_file, regex, headers)
    print("Log converting to dataframe completed.")
    rex = [
        r'[^A-Za-z]',  # remove non-alphabetic characters
        r' +'  # remove multiple Spaces
    ]
    parsed_log_messages = []
    print("Proprocessing log...")
    for idx, line in tqdm(df_log.iterrows(), total=df_log.shape[0]):
        line_content = line['Content']
        logmessage = line_content
        for currentRex in rex:
            logmessage = re.sub(currentRex, ' ', logmessage)
        logmessage = logmessage.strip().lower()  # 去除前后的空格
        # print(logmessage)
        parsed_log_messages.append(logmessage)

    df_log['EventTemplate'] = parsed_log_messages
    df_log.to_csv(log_structured_file, index=False)
    print("Log preprocessing completed.")


def hdfs_sampling(log_structured_file, log_sequence_file, window='session'):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_structured_file, "for sampling...")
    df = pd.read_csv(log_structured_file, engine='c',
                     na_filter=False, memory_map=True, dtype={'Date': object, "Time": object})
    # df = df.head(400000)
    print("Sampling...")
    data_dict = defaultdict(str)  # preserve insertion order of items
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if data_dict[blk_Id] == '':
                data_dict[blk_Id] = data_dict[blk_Id] + row["EventTemplate"]
            else:
                data_dict[blk_Id] = data_dict[blk_Id] + '. ' + row["EventTemplate"]

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    data_df.to_csv(log_sequence_file, index=False)
    print("HDFS sampling completed.")


def generate_sequence_label(log_sequence_file, log_sequence_label_file):
    blk_label_dict = {}
    blk_label_file = os.path.join(input_dir, "anomaly_label.csv")
    print("Loading", blk_label_file, "for generating labels...")
    blk_df = pd.read_csv(blk_label_file)
    print("Generating labels...")
    for _, row in tqdm(blk_df.iterrows(), total=blk_df.shape[0]):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    seq = pd.read_csv(log_sequence_file)
    seq["Label"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x))
    seq.to_csv(log_sequence_label_file, index=False)
    print("Labels generating completed.")


def generate_train_test_byratio(log_sequence_label_file, ratio, shuffle, test_small_ratio):
    seq = pd.read_csv(log_sequence_label_file)
    if shuffle == True:
        seq = seq.sample(frac=1, random_state=args.seed)
    train_len = int(len(seq) * ratio)
    train = seq.iloc[:train_len][['BlockId', 'EventSequence', 'Label']]
    train = train.sample(frac=1, random_state=args.seed)
    test = seq.iloc[train_len:][['BlockId', 'EventSequence', 'Label']]
    test = test.sample(frac=1, random_state=args.seed)

    print("training size {0}, training size normal {1}, training size abnormal {2}".format(len(train), len(train[train["Label"] == 0]),
                                                                                           len(train[train["Label"] == 1])))
    print("testing size {0}, testing size normal {1}, testing size abnormal {2}".format(len(test), len(test[test["Label"] == 0]),
                                                                                        len(test[test["Label"] == 1])))
    training_set_path = os.path.join(output_dir, "train" + str(ratio) + ".csv")
    train.to_csv(training_set_path, index=False)
    testing_set_path = os.path.join(output_dir, "test" + str(round(1 - ratio, 1)) + ".csv")
    test.to_csv(testing_set_path, index=False)
    if test_small_ratio + ratio != 1:
        test_small = test.iloc[:int((test_small_ratio / (1 - ratio)) * len(test))][['BlockId', 'EventSequence', 'Label']]
        testing_small_set_path = os.path.join(output_dir, "test" + str(round(test_small_ratio, 2)) + ".csv")
        test_small.to_csv(testing_small_set_path, index=False)
    print("Splitting training and testing set completed.")
    return training_set_path, testing_set_path


def get_file_for_DAPT(train_set_path, train_steps, batch_size):
    seq = pd.read_csv(train_set_path)

    if train_steps != None:
        num_of_lines = train_steps * batch_size
        train_file = open(os.path.join(output_dir, 'DAPTtrain' + str(train_steps) + '_' + str(batch_size) + '.txt'),
                          mode='a')
    else:
        num_of_lines = len(seq)
        train_file = open(os.path.join(output_dir, 'DAPTtrain.txt'), mode='a')
    train_file.seek(0)
    train_file.truncate()

    if num_of_lines / len(seq) < 1:
        seq = seq.sample(frac=num_of_lines / len(seq), random_state=args.seed)
    else:
        seq = seq.sample(frac=1, random_state=args.seed)
    for i, row in tqdm(seq.iterrows(), total=seq.shape[0]):
        train_file.write(row["EventSequence"] + "\n")
    train_file.close()
    print("Got DAPT training file.")


def generate_train_small_sample(train_file, small_train_len, small_train_len_normal):
    seq = pd.read_csv(train_file)
    seq = seq.sample(frac=1, random_state=args.seed)
    normal_seq = seq[seq["Label"] == 0]
    # normal_seq = normal_seq.sample(frac=1, random_state=args.seed)  # shuffle normal data

    abnormal_seq = seq[seq["Label"] == 1]
    # abnormal_seq = abnormal_seq.sample(frac=1, random_state=args.seed)  # shuffle abnormal data
    train_len_normal, train_len_abnormal = len(normal_seq), len(abnormal_seq)
    print("train normal size {0}, train abnormal size {1}".format(train_len_normal, train_len_abnormal))

    small_train_len_abnormal = small_train_len - small_train_len_normal  # 训练集中异常的日志数量

    small_train_normal = normal_seq.iloc[:small_train_len_normal][['BlockId', 'EventSequence', 'Label']]
    small_train_abnormal = abnormal_seq.iloc[:small_train_len_abnormal][['BlockId', 'EventSequence', 'Label']]
    train = pd.concat([small_train_normal, small_train_abnormal], axis=0)

    train.to_csv(os.path.join(output_dir, "train" + str(small_train_len_normal) + '+' + str(small_train_len_abnormal) + ".csv"),
                 index=False)
    print("Got anomaly detection training file.")


def generate_test_sample(test_file, test_small_ratio, anomaly_percentage):
    seq = pd.read_csv(test_file)
    normal_seq = seq[seq["Label"] == 0]
    abnormal_seq = seq[seq["Label"] == 1]
    length = int(len(seq) * test_small_ratio / 0.2)
    ab = int((anomaly_percentage / 100) * len(seq) * (test_small_ratio / 0.2))
    nor = length - ab
    small_test_normal = normal_seq.iloc[:nor][['BlockId', 'EventSequence', 'Label']]
    small_test_abnormal = abnormal_seq.iloc[:ab][['BlockId', 'EventSequence', 'Label']]
    test = pd.concat([small_test_normal, small_test_abnormal], axis=0)
    test.to_csv(os.path.join(output_dir, "test" + str(test_small_ratio) + '_' + str(anomaly_percentage) + ".csv"), index=False)


if __name__ == "__main__":
    # 1. Preprocess HDFS log
    print("1. Preprocess HDFS log")
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    preprocess(input_file, log_structured_file, log_format)

    # 2. Sample HDFS log into sequences according to block_id
    print("\n2. Sample HDFS log into sequences according to block_id")
    hdfs_sampling(log_structured_file, log_sequence_file, window='session')

    # 3. Generate anomaly labels
    print("\n3. Generate anomaly labels")
    generate_sequence_label(log_sequence_file, log_sequence_label_file)

    # 4. Split training and testing set
    print("\n4. Split training and testing set")
    training_set_path, testing_set_path = generate_train_test_byratio(log_sequence_label_file, ratio=args.train_ratio,
                                                                      shuffle=args.shuffle,
                                                                      test_small_ratio=args.test_small_ratio)
    # training_set_path = output_dir + "/train0.8.csv"
    # 5. Get DAPT training file
    print("\n5. Get DAPT training file")
    get_file_for_DAPT(train_set_path=training_set_path, train_steps=args.dapt_train_steps, batch_size=args.dapt_batch_size)

    # 6. Get anomaly detection training file
    print("\n6. Get anomaly detection training file")
    generate_train_small_sample(train_file=training_set_path, small_train_len=args.small_train_len,
                                small_train_len_normal=args.small_train_len_normal)

    # This function is to generate testing file with different anomaly ratio.
    # generate_test_sample(test_file=testing_set_path, test_small_ratio=0.05, anomaly_percentage=0.5)
