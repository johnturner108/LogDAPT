import sys

sys.path.append('../')

import os
import pandas as pd
import argparse
from tqdm import tqdm
import re
from utils import str2bool

tqdm.pandas()
pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument("-input_dir", default='../../datasets/bgl/')
parser.add_argument("-output_dir", default='../../preprocessed_data/bgl/')
parser.add_argument("-log_file", default='BGL.log')
parser.add_argument("-train_ratio", default=0.8, type=float)
parser.add_argument("-test_small_ratio", default=0.01, type=float)
parser.add_argument("-shuffle", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-small_train_len", default=2000, type=int)
parser.add_argument("-small_train_len_normal", default=1000, type=int)
parser.add_argument("-dapt_train_steps", default=5000, type=int)
parser.add_argument("-dapt_batch_size", default=16, type=int)
parser.add_argument("-eval_size", default=5000, type=int)
parser.add_argument("-seed", default=21, type=int)
parser.add_argument("-step_size", default=1, type=int)
parser.add_argument("-window_size", default=20, type=int)

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir  # The output directory of preprocessing results
log_file = args.log_file  # The input log file name
input_file = os.path.join(input_dir, log_file)
log_structured_file = os.path.join(output_dir, log_file + "_structured.csv")
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
            for line in tqdm(fin.readlines()):
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1  # 总共的日志信息共4747963，符合regex的为4713493，其他日志信息没有Content列
                except Exception as e:
                    # print("\n", line)
                    # print(e)
                    pass
        print("Total size of logs is", linecount, 'out of', cnt, end='.\n')
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    print("Converting log to dataframe...")
    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(input_file, regex, headers)
    # Total size after encoding is 4713493 out of 4747963
    print("Log converting to dataframe completed.")
    rex = [
        r'(0x)[0-9a-fA-F]+',  # hexadecimal
        r"(/[-\w]+)+",  # file path
        r'\d+\.\d+\.\d+\.\d+',  # IP
        r'([0-9A-F]{2}:)+([0-9A-F]){2}',  # remove FF:F2:9F:16:C3:AD:00:0D:60:E9:3C:52
        r'([A-z]+[\d]+)|([\d]+[A-z]+)',  # remove 04C480A70C2FFFFF0A081A50CAC5, U01, J11
        r'[^A-Za-z]',  # remove non-alphabetic characters
        r' +',  # remove multiple Spaces
    ]
    parsed_log_messages = []
    print("Proprocessing log...")
    for idx, line in tqdm(df_log.iterrows(), total=df_log.shape[0]):
        line_content = line['Content']
        for currentRex in rex:
            # message = re.sub(currentRex, ' ', line_content)
            # message = re.sub(r' +', ' ', message)
            # if message != ' ':
            line_content = re.sub(currentRex, ' ', line_content)
        line_content = line_content.strip().lower()  # 去除前后的空格
        parsed_log_messages.append(line_content)

    df_log['EventTemplate'] = parsed_log_messages
    df_log.to_csv(log_structured_file, index=False)
    print("Log preprocessing completed.")


def sampling_fixed_window(log_structured_file, log_sequence_label_file, step_size, window_size):
    print("Loading", log_structured_file, "for sampling...")
    df = pd.read_csv(log_structured_file)
    # df = df.head(200000)
    log_size = len(df)
    # In the first column of the log, "-" indicates non-alert messages while others are alert messages.
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    start_index = 0
    end_index = window_size
    new_data = []
    num_session = 1
    print("Sampling BGL log...")
    bar = tqdm(total=(log_size - window_size) // step_size)
    while end_index < log_size:
        event_sequence = ''
        i = 0
        for k in range(start_index, end_index):
            if i == 0:
                event_sequence = event_sequence + str(df["EventTemplate"][k])
            else:
                event_sequence = event_sequence + ' ' + str(df["EventTemplate"][k])
            i += 1
        new_data.append([
            event_sequence,
            max(df["Label"][start_index:end_index]),
        ])
        start_index += step_size
        end_index += step_size
        # if num_session % 1000 == 0:
        #     print("process {} window".format(num_session), end='\n')
        num_session += 1
        bar.update(1)
    bar.close()
    new_df = pd.DataFrame(new_data, columns=["EventSequence", "Label"])
    new_df.to_csv(log_sequence_label_file, index=False)
    print('BGl sampling completed. There are %d instances (fixed windows) in this dataset\n' % len(new_df))


def generate_train_test_byratio(log_sequence_label_file, ratio, shuffle, test_small_ratio):
    seq = pd.read_csv(log_sequence_label_file)
    if shuffle == True:
        seq = seq.sample(frac=1, random_state=args.seed)
    train_len = int(len(seq) * ratio)
    train = seq.iloc[:train_len][['EventSequence', 'Label']]
    train = train.sample(frac=1, random_state=args.seed)
    test = seq.iloc[train_len:][['EventSequence', 'Label']]
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
        test_small = test.iloc[:int((test_small_ratio / (1 - ratio)) * len(test))][['EventSequence', 'Label']]
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

    small_train_normal = normal_seq.iloc[:small_train_len_normal][['EventSequence', 'Label']]
    small_train_abnormal = abnormal_seq.iloc[:small_train_len_abnormal][['EventSequence', 'Label']]
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
    small_test_normal = normal_seq.iloc[:nor][['EventSequence', 'Label']]
    small_test_abnormal = abnormal_seq.iloc[:ab][['EventSequence', 'Label']]
    test = pd.concat([small_test_normal, small_test_abnormal], axis=0)
    test.to_csv(output_dir + "test" + str(test_small_ratio) + '_' + str(anomaly_percentage) + ".csv", index=False)


if __name__ == "__main__":
    # 1. Preprocess BGL log
    print("1. Preprocess BGL log")
    log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    preprocess(input_file, log_structured_file, log_format)

    # 2. Sample BGL log into sequences with fixed sliding window, obtaining labels at the same time
    print("\n2. Sample BGL log into sequences with fixed sliding window, obtaining labels at the same time")
    sampling_fixed_window(log_structured_file, log_sequence_label_file, step_size=args.step_size, window_size=args.window_size)

    # 3. Split training and testing set
    print("\n3. Split training and testing set")
    training_set_path, testing_set_path = generate_train_test_byratio(log_sequence_label_file, ratio=args.train_ratio,
                                                                      shuffle=args.shuffle,
                                                                      test_small_ratio=args.test_small_ratio)

    # 4. Get DAPT training file
    print("\n4. Get DAPT training file")
    get_file_for_DAPT(train_set_path=training_set_path, train_steps=args.dapt_train_steps, batch_size=args.dapt_batch_size)

    # 5. Get anomaly detection training file
    print("\n5. Get anomaly detection training file")
    generate_train_small_sample(train_file=training_set_path, small_train_len=args.small_train_len,
                                small_train_len_normal=args.small_train_len_normal)

    # This function is to generate testing file with different anomaly ratio.
    # generate_test_sample(test_file=testing_set_path, test_small_ratio=0.01, anomaly_percentage=0.1)
