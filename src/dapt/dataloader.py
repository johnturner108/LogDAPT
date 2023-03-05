from torch.utils.data import Dataset
from transformers_modified.tokenization_utils import PreTrainedTokenizer
import os
import torch
from transformers_modified.utils import logging
from typing import Dict, List, Optional
from transformers_modified.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import math

logger = logging.get_logger(__name__)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    span: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        # if self.mlm:
        #     batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
        #         batch["input_ids"], special_tokens_mask=special_tokens_mask
        #     )
        if self.span:
            batch["input_ids"], batch["labels"], batch["targets"], batch["pairs"] = self.span_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        # inputs和labels都是Tensor: (batch_size, sentence_length)
        return inputs, labels

    def span_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch
        p = 0.2
        lower = 1
        upper = 10
        max_pair_targets = 15
        lens = []
        for i in range(lower, upper + 1):
            lens.append(i)
        len_distrib = [p * (1 - p) ** (i - lower) for i in
                       range(lower, upper + 1)] if p >= 0 else None
        len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
        sentences = inputs.detach().numpy()
        inputs = np.empty([inputs.shape[0], inputs.shape[1]], dtype=int)
        labels = np.empty([inputs.shape[0], inputs.shape[1]], dtype=int)
        batch_targets = []
        batch_pairs = []
        max_pair_num = 0
        for i in range(len(sentences)):
            sentence, target, pair_targets, pairs = self.span_mask_a_sentence(sentences[i], lens, len_distrib, max_pair_targets)
            inputs[i] = sentence
            labels[i] = target
            batch_targets.append(pair_targets)
            batch_pairs.append(pairs)
            if len(pairs) > max_pair_num:
                max_pair_num = len(pairs)
        for i in range(len(batch_pairs)):
            for j in range(max_pair_num - len(batch_pairs[i])):
                batch_pairs[i].append([0, 0])
            for j in range(max_pair_num - len(batch_targets[i])):
                batch_targets[i].append([-100 for _ in range(max_pair_targets)])
        inputs = torch.from_numpy(inputs)  # ndarray转换为tensor
        labels = torch.from_numpy(labels)  # ndarray转换为tensor
        batch_targets = torch.LongTensor(batch_targets)
        batch_pairs = torch.LongTensor(batch_pairs)
        return inputs, labels, batch_targets, batch_pairs

    def span_mask_a_sentence(self, sentence, lens, len_distrib, max_pair_targets):
        ## 要减去[CLS]和[SEP]
        ## [0, 1, 2, 3, , ,113 ,114, 126, 127]
        ## 先找到[SEP]的序号
        sep_position = len(sentence)  ##随便赋值一下
        for i in reversed(range(len(sentence))):
            if sentence[i] == self.tokenizer.sep_token_id:
                sep_position = i
        mask_num = math.ceil((sep_position - 1) * self.mlm_probability)
        # token = self.tokenizer.decode(sentence[8])
        # print(token.startswith("# #"))
        mask = set()
        spans = []
        while len(mask) < mask_num:
            span_len = np.random.choice(lens, p=len_distrib)
            # anchor  = np.random.choice(sent_length)
            anchor = np.random.randint(1, sep_position)  ## 从[CLS]后的第一个词开始到[SEP]的前一个词
            if anchor in mask:  ## 如果选到的span的开始（即anchor）在mask里面，那么重新选anchor
                continue
            # find word start, end
            left1, right1 = self.get_word_start(sentence, anchor), self.get_word_end(sentence, anchor)
            spans.append([left1, left1])
            for i in range(left1, right1):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i
            num_words = 1
            right2 = right1
            while num_words < span_len and right2 < sep_position and len(mask) < mask_num:
                # complete current word  ## 如果当前span加入mask的词小于span的长度、span的最右边的词小于sentence的长度、mask集合里的token数小于要mask的token数
                left2 = right2
                right2 = self.get_word_end(sentence, right2)
                num_words += 1
                for i in range(left2, right2):
                    if len(mask) >= mask_num:
                        break
                    mask.add(i)
                    spans[-1][-1] = i
        pad = -100
        tokens = []  ## 整个数据集的token
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        sentence, target, pair_targets = self.span_masking(sentence, spans, tokens, pad, mask_id, max_pair_targets,
                                                           mask, replacement='span',
                                                           endpoints='external')
        # if self.args.return_only_spans:
        #     pair_targets = None
        return sentence, target, pair_targets, spans

    def get_word_start(self, sentence, anchor):
        left = anchor
        while left > 0 and self.tokenizer.decode(sentence[left]).startswith("# #") == True:
            left -= 1
        return left

    # word end is next word start
    def get_word_end(self, sentence, anchor):
        right = anchor + 1
        while right < len(sentence) and self.tokenizer.decode(sentence[right]).startswith("# #") == True:
            right += 1
        return right

    def span_masking(self, sentence, spans, tokens, pad, mask_id, pad_len, mask, replacement='word_piece', endpoints='external'):
        ## tokens暂时没用，不知道怎么获取
        sentence = np.copy(sentence)
        sent_length = len(sentence)
        target = np.full(sent_length, pad)
        pair_targets = []
        spans = merge_intervals(spans)
        assert len(mask) == sum([e - s + 1 for s, e in spans])
        # print(list(enumerate(sentence)))
        for start, end in spans:
            lower_limit = 0 if endpoints == 'external' else -1
            upper_limit = sent_length - 1 if endpoints == 'external' else sent_length
            if start > lower_limit and end < upper_limit:
                if endpoints == 'external':
                    ## 不要span的两边token的索引了
                    # pair_targets += [[start - 1, end + 1]]
                    pair_targets += [[]]
                    pass
                else:
                    # pair_targets += [[start, end]]
                    pair_targets += []
                    pass
                # pair_targets[-1] += [sentence[i] for i in range(start, end + 1)]
                pair_targets[-1] += [sentence[i] for i in range(start, end + 1)]
            rand = np.random.random()
            for i in range(start, end + 1):
                assert i in mask
                target[i] = sentence[i]
                if replacement == 'word_piece':
                    rand = np.random.random()
                if rand < 0.8:
                    sentence[i] = mask_id
                elif rand < 0.9:
                    # self.tokenizer
                    # sample random token according to input distribution
                    # sentence[i] = np.random.choice(tokens)
                    sentence[i] = np.random.randint(len(self.tokenizer.get_vocab()))
        pair_targets = pad_to_len(pair_targets, pad, pad_len)
        # if pair_targets is None:
        return sentence, target, pair_targets


def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def pad_to_len(pair_targets, pad, max_pair_target_len):
    for i in range(len(pair_targets)):
        pair_targets[i] = pair_targets[i][:max_pair_target_len]
        this_len = len(pair_targets[i])
        for j in range(max_pair_target_len - this_len):
            pair_targets[i].append(pad)
    return pair_targets


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result
