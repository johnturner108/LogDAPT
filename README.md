# LogDAPT

**This code is for paper [LogDAPT: Industrial Log Data Anomaly Detection with Domain-Adaptive Pretraining]()**


Results on HDFS, BGL and Thunderbird:


<table class="tg">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">Neural Log</th>
    <th class="tg-0pky">BERT base</th>
    <th class="tg-0pky">LogDAPT (MLM)</th>
    <th class="tg-0pky">LogDAPT (Span)</th>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="5">HDFS</td>
  </tr>
  <tr>
    <td class="tg-0pky">Precision</td>
    <td class="tg-0pky">0.96</td>
    <td class="tg-0pky">0.8944</td>
    <td class="tg-0pky">0.9381</td>
    <td class="tg-0pky">0.9432</td>
  </tr>
  <tr>
    <td class="tg-0pky">Recall</td>
    <td class="tg-0pky">1.0</td>
    <td class="tg-0pky">1.0</td>
    <td class="tg-0pky">1.0</td>
    <td class="tg-0pky">1.0</td>
  </tr>
  <tr>
    <td class="tg-0pky">F1-Score</td>
    <td class="tg-0pky">0.98</td>
    <td class="tg-0pky">0.9443</td>
    <td class="tg-0pky">0.9681</td>
    <td class="tg-0pky">0.9708</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="5">BGL</td>
  </tr>
  <tr>
    <td class="tg-0lax">Precision</td>
    <td class="tg-0lax">0.98</td>
    <td class="tg-0lax">0.9022</td>
    <td class="tg-0lax">0.9678</td>
    <td class="tg-0lax">0.9503</td>
  </tr>
  <tr>
    <td class="tg-0lax">Recall</td>
    <td class="tg-0lax">0.98</td>
    <td class="tg-0lax">0.9812</td>
    <td class="tg-0lax">0.9782</td>
    <td class="tg-0lax">0.9918</td>
  </tr>
  <tr>
    <td class="tg-0lax">F1-Score</td>
    <td class="tg-0lax">0.98</td>
    <td class="tg-0lax">0.9401</td>
    <td class="tg-0lax">0.973</td>
    <td class="tg-0lax">0.9706</td>
  </tr>
  <td class="tg-baqh" colspan="5">Thunderbird</td>
  <tr>
    <td class="tg-0lax">Precision</td>
    <td class="tg-0lax">0.93</td>
    <td class="tg-0lax">0.9009</td>
    <td class="tg-0lax">0.9896</td>
    <td class="tg-0lax">0.9921</td>
  </tr>
  <tr>
    <td class="tg-0lax">Recall</td>
    <td class="tg-0lax">1.0</td>
    <td class="tg-0lax">1.0</td>
    <td class="tg-0lax">1.0</td>
    <td class="tg-0lax">1.0</td>
  </tr>
  <tr>
    <td class="tg-0lax">F1-Score</td>
    <td class="tg-0lax">0.96</td>
    <td class="tg-0lax">0.9479</td>
    <td class="tg-0lax">0.9948</td>
    <td class="tg-0lax">0.996</td>
  </tr>
</table>

**Python version**: This code is in Python3.9

**Package Requirements**: torch==1.13.1 transformers pandas tqdm scikit-learn datasets matplotlib

Some codes are borrowed from LogBERT(https://github.com/HelenGuohx/logbert) and SpanBERT(https://github.com/facebookresearch/SpanBERT)

## Trained Models
[LogDAPT_HDFS_MLM](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

[LogDAPT_HDFS_Span](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

[LogDAPT_BGL_MLM](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

[LogDAPT_BGL_Span](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

[LogDAPT_Thunderbird_MLM](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

[LogDAPT_Thunderbird_Span](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

## Preprocessed Data
[Preprocessed data]()

## Data Preprocessing

**Use the `.py` files under `src/preprocessing` to preprocess the log data**
### 1. HDFS
```
python data_process_hdfs.py -input_dir INPUT_DATA_PATH -output_dir OUTPUT_DATA_PATH -log_file LOG_FILE_NAME -train_ratio 0.8 -shuffle true -small_train_len 2000 -small_train_len_normal 1000 -dapt_train_steps 5000 -dapt_batch_size 16 -eval_size 5000 -seed 21
```
* `INPUT_DATA_PATH` is the directory for raw logs. It could be `../../datasets/hdfs/`, if you put the log file `HDFS.log` inside the directory `LogDAPT/datasets`

* `OUTPUT_DATA_PATH` is the directory of preprocessed log data, including DAPT training files, anomaly detection training files and testing files. For example: `../../preprocessed_data/hdfs/`
* For HDFS dataset, `LOG_FILE_NAME` is normally `HDFS.log` if you didn't change it.
* `-train_ratio` is the ratio for splitting training set and testing set, the actual training samples are only a small part of the training set.
* If `-shuffle` is `true`, it means shuffling is applied before splitting the dataset into training set and testing set, so that the training set contains part of the future data and the testing set contains part of the past data.
* `-small_train_len` is the number of samples for downstream anomaly detection training, and `small_train_len_normal` is the number of normal sequences inside.
* `-dapt_train_steps` is how many steps you want to perform Domain_Adaptive PreTraining (DAPT), and `dapt_batch_size` is the batch size for DAPT. These two parameters exist to prevent unnecessary encoding in DAPT training stage in case you don't want to train for so many steps.
* You can specify `-test_small_ratio` to a smaller percentage less than `1 - train_ratio` such as `0.05` if you find testing takes too long.

### 2. BGL
```
python data_process_bgl.py -input_dir INPUT_DATA_PATH -output_dir OUTPUT_DATA_PATH -log_file LOG_FILE_NAME -train_ratio 0.8 -shuffle true -small_train_len 2000 -small_train_len_normal 1000 -dapt_train_steps 5000 -dapt_batch_size 16 -eval_size 5000 -seed 21 -step_size 1 -window_size 20
```

### 3. Thunderbird
```
python data_process_tbird.py -input_dir INPUT_DATA_PATH -output_dir OUTPUT_DATA_PATH -log_file LOG_FILE_NAME -train_ratio 0.8 -shuffle true -small_train_len 400 -small_train_len_normal 200 -dapt_train_steps 5000 -dapt_batch_size 16 -eval_size 5000 -seed 21 -step_size 1 -window_size 20
```
* We set `-step_size` to `1`, and `-window_size` to `20` for both BGL and Thunderbird dataset according to ASE 2021 paper [Log-based Anomaly Detection Without Log Parsing]()

## DAPT Training

### 1 MLM
```
python domain_pretraining_mlm.py -dataset DATASET_NAME -train_file TRAIN_FILE_PATH -model_output MODEL_PATH -block_size 512 -mlm_probability 0.15 -learning_rate 5e-5 -visible_gpus 0,1 -num_train_epochs 1 -max_steps -1 -save_steps 500 -train_batch_size 16
```
* The choices of `DATASET_NAME` are `hdfs`, `bgl` and `tbird`, make sure that `DATASET_NAME` corresponds to `TRAIN_FILE_PATH` and `MODEL_PATH`
* `TRAIN_FILE_PATH` is the DAPTtrain.txt file produced by data preprocessing, for example `../../preprocessed_data/hdfs/DAPTtrain5000_16.txt`
* `MODEL_PATH` is the path where you want to save your models, for example `LogDAPT/models/hdfs`
* `-block_size` is the maximum number of input tokens you wish the model to process.
* `-mlm_probability` is the percentage of tokens that will be masked.
* `-max_steps` is the number of steps you wish the model be trained for, we trained the model for 5000 steps in our paper.
* Checkpoints are saved per `save_steps` steps.

### 2 Span
```
python domain_pretraining_span.py -dataset DATASET_NAME -train_file TRAIN_FILE_PATH -model_output MODEL_PATH -block_size 512 -mlm_probability 0.15 -learning_rate 5e-5 -visible_gpus 0,1 -num_train_epochs 1 -max_steps -1 -save_steps 500 -train_batch_size 16
```
* Hyper parameters of Span pretraining strategy is specified in `src/dapt/dataloader.py` according to the ACL 2020 paper [SpanBERT: Improving Pre-training by Representing and Predicting Spans]()


## Anomaly Training and Evaluation
### Training
```
python train.py -dataset DATASET_NAME -do_train -pretrained_model PRETRAINED_MODEL_PATH -train_file TRAIN_FILE_PATH -test_file TEST_FILE_PATH -train_batch_size 32 -eval_batch_size 64 -test_batch_size 32 -learning_rate 5e-5 -train_epochs 2 -train_show_step 50 -test_show_step 50 -visible_gpu 0 -seed 43
```

* The choices of `DATASET_NAME` are `hdfs`, `bgl` and `tbird`, make sure that `DATASET_NAME` corresponds to `PRETRAINED_MODEL_PATH `, `TRAIN_FILE_PATH` and `TEST_FILE_PATH`
* If `PRETRAINED_MODEL_PATH` is specified, trained model will be automatically save in `PRETRAINED_MODEL_PATH`.
### Testing
```
python train.py -dataset DATASET_NAME -do_test -saved_model SAVED_MODEL_PATH -test_file TEST_FILE_PATH -test_batch_size 32 -test_show_step 50 -visible_gpu 0 -seed 43
```
* `SAVED_MODEL_PATH` is the directory of saved anomaly detection model.

