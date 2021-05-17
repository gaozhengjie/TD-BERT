import tokenization_word as tokenization
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np

# 读取数据集，并预处理
class ReadData:
    def __init__(self, opt):
        print("load data ...")
        self.opt = opt
        self.train_examples = opt.processor.get_train_examples(opt.data_dir)
        self.eval_examples = opt.processor.get_dev_examples(opt.data_dir)
        self.label_list = opt.processor.get_labels()

        self.tokenizer = tokenization.FullTokenizer(vocab_file=opt.vocab_file, do_lower_case=opt.do_lower_case)
        self.train_dataloader = self.get_data_loader(examples=self.train_examples, type='train_data')
        self.eval_dataloader = self.get_data_loader(examples=self.eval_examples, type='eval_data')

    def get_data_loader(self, examples, type='train_data'):
        features = self.convert_examples_to_features(
            examples, self.label_list, self.opt.max_seq_length, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_input_t_ids = torch.tensor([f.input_t_ids for f in features], dtype=torch.long)
        all_input_t_mask = torch.tensor([f.input_t_mask for f in features], dtype=torch.long)
        all_segment_t_ids = torch.tensor([f.segment_t_ids for f in features], dtype=torch.long)

        all_input_without_t_ids = torch.tensor([f.input_without_t_ids for f in features], dtype=torch.long)
        all_input_without_t_mask = torch.tensor([f.input_without_t_mask for f in features], dtype=torch.long)
        all_segment_without_t_ids = torch.tensor([f.segment_without_t_ids for f in features], dtype=torch.long)

        all_input_left_t_ids = torch.tensor([f.input_left_t_ids for f in features], dtype=torch.long)
        all_input_left_t_mask = torch.tensor([f.input_left_t_mask for f in features], dtype=torch.long)
        all_segment_left_t_ids = torch.tensor([f.segment_left_t_ids for f in features], dtype=torch.long)

        all_input_right_t_ids = torch.tensor([f.input_right_t_ids for f in features], dtype=torch.long)
        all_input_right_t_mask = torch.tensor([f.input_right_t_mask for f in features], dtype=torch.long)
        all_segment_right_t_ids = torch.tensor([f.segment_right_t_ids for f in features], dtype=torch.long)

        input_left_ids = torch.tensor([f.input_left_ids for f in features], dtype=torch.long)
        input_left_mask = torch.tensor([f.input_left_mask for f in features], dtype=torch.long)
        segment_left_ids = torch.tensor([f.segment_left_ids for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_t_ids,
                             all_input_t_mask, all_segment_t_ids, all_input_without_t_ids, all_input_without_t_mask,
                             all_segment_without_t_ids, all_input_left_t_ids, all_input_left_t_mask, all_segment_left_t_ids,
                             all_input_right_t_ids, all_input_right_t_mask, all_segment_right_t_ids,
                             input_left_ids, input_left_mask, segment_left_ids)
        if type == 'train_data':
            train_data = data
            train_sampler = RandomSampler(data)
            return DataLoader(train_data, sampler=train_sampler, batch_size=self.opt.train_batch_size)
        else:
            eval_data = data
            eval_sampler = SequentialSampler(eval_data)
            return DataLoader(eval_data, sampler=eval_sampler, batch_size=self.opt.eval_batch_size)

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_aspect = tokenizer.tokenize(example.aspect)
            tokens_text_without_target = tokenizer.tokenize(example.text_without_target)
            tokens_text_left_with_target = tokenizer.tokenize(example.text_left_with_target)
            tokens_text_right_with_target = tokenizer.tokenize(example.text_right_with_target)
            tokens_text_left = tokenizer.tokenize(example.text_left)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_aspect:  # 如果不为 None
                tokens_t = []
                segment_t_ids = []
                tokens_t.append("[CLS]")
                segment_t_ids.append(0)
                for token in tokens_aspect:
                    tokens_t.append(token)
                    segment_t_ids.append(0)
                tokens_t.append("[SEP]")
                segment_t_ids.append(0)
                input_t_ids = tokenizer.convert_tokens_to_ids(tokens_t)
                input_t_mask = [1] * len(input_t_ids)
                while len(input_t_ids) < max_seq_length:
                    input_t_ids.append(0)
                    input_t_mask.append(0)
                    segment_t_ids.append(0)
                assert len(input_t_ids) == max_seq_length
                assert len(input_t_mask) == max_seq_length
                assert len(segment_t_ids) == max_seq_length
                # 以下是处理句子中剔除target词的情况，tokens_text_without_target
                tokens_without_target = []
                segment_without_t_ids = []
                tokens_without_target.append("[CLS]")
                segment_without_t_ids.append(0)
                for token in tokens_text_without_target:
                    tokens_without_target.append(token)
                    segment_without_t_ids.append(0)
                tokens_without_target.append("[SEP]")
                segment_without_t_ids.append(0)
                input_without_t_ids = tokenizer.convert_tokens_to_ids(tokens_without_target)
                input_without_t_mask = [1] * len(input_without_t_ids)
                while len(input_without_t_ids) < max_seq_length:
                    input_without_t_ids.append(0)
                    input_without_t_mask.append(0)
                    segment_without_t_ids.append(0)
                assert len(input_without_t_ids) == max_seq_length
                assert len(input_without_t_mask) == max_seq_length
                assert len(segment_without_t_ids) == max_seq_length
                # 以下是处理目标词左侧的句子，含目标词 tokens_text_left_with_aspect
                tokens_left_target = []
                segment_left_t_ids = []
                tokens_left_target.append("[CLS]")
                segment_left_t_ids.append(0)
                for token in tokens_text_left_with_target:
                    tokens_left_target.append(token)
                    segment_left_t_ids.append(0)
                tokens_left_target.append("[SEP]")
                segment_left_t_ids.append(0)
                input_left_t_ids = tokenizer.convert_tokens_to_ids(tokens_left_target)
                input_left_t_mask = [1] * len(input_left_t_ids)
                while len(input_left_t_ids) < max_seq_length:
                    input_left_t_ids.append(0)
                    input_left_t_mask.append(0)
                    segment_left_t_ids.append(0)
                assert len(input_left_t_ids) == max_seq_length
                assert len(input_left_t_mask) == max_seq_length
                assert len(segment_left_t_ids) == max_seq_length
                # 以下是处理目标词右侧的句子，含目标词 tokens_text_right_with_aspect
                tokens_right_target = []
                segment_right_t_ids = []
                tokens_right_target.append("[CLS]")
                segment_right_t_ids.append(0)
                for token in tokens_text_right_with_target:
                    tokens_right_target.append(token)
                    segment_right_t_ids.append(0)
                tokens_right_target.append("[SEP]")
                segment_right_t_ids.append(0)
                input_right_t_ids = tokenizer.convert_tokens_to_ids(tokens_right_target)
                input_right_t_mask = [1] * len(input_right_t_ids)
                while len(input_right_t_ids) < max_seq_length:
                    input_right_t_ids.append(0)
                    input_right_t_mask.append(0)
                    segment_right_t_ids.append(0)
                assert len(input_right_t_ids) == max_seq_length
                assert len(input_right_t_mask) == max_seq_length
                assert len(segment_right_t_ids) == max_seq_length
                # 以下是处理目标词右侧的句子，不包含目标词 tokens_text_left
                tokens_left = []
                segment_left_ids = []
                tokens_left.append("[CLS]")
                segment_left_ids.append(0)
                for token in tokens_text_left:
                    tokens_left.append(token)
                    segment_left_ids.append(0)
                tokens_left.append("[SEP]")
                segment_left_ids.append(0)
                input_left_ids = tokenizer.convert_tokens_to_ids(tokens_left)
                input_left_mask = [1] * len(input_left_ids)
                while len(input_left_ids) < max_seq_length:
                    input_left_ids.append(0)
                    input_left_mask.append(0)
                    segment_left_ids.append(0)
                assert len(input_left_ids) == max_seq_length
                assert len(input_left_mask) == max_seq_length
                assert len(segment_left_ids) == max_seq_length


            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]
            # if ex_index < 5:
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s" % (example.guid))
            #     logger.info("tokens: %s" % " ".join(
            #         [tokenization.printable_text(x) for x in tokens]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     logger.info("label: %s (id = %d)" % (example.label, label_id))

            if tokens_aspect == None:
                features.append(
                    InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id, ))
            else:
                features.append(
                    InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        input_t_ids=input_t_ids,
                        input_t_mask=input_t_mask,
                        segment_t_ids=segment_t_ids,
                        input_without_t_ids=input_without_t_ids,
                        input_without_t_mask=input_without_t_mask,
                        segment_without_t_ids=segment_without_t_ids,
                        input_left_t_ids=input_left_t_ids,
                        input_left_t_mask=input_left_t_mask,
                        segment_left_t_ids=segment_left_t_ids,
                        input_right_t_ids=input_right_t_ids,
                        input_right_t_mask=input_right_t_mask,
                        segment_right_t_ids=segment_right_t_ids,
                        input_left_ids=input_left_ids,
                        input_left_mask=input_left_mask,
                        segment_left_ids=segment_left_ids,
                ))
        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        file_in = open(input_file, "rb")
        lines = []
        for line in file_in:
            lines.append(line.decode("utf-8").split("\t"))
        return lines



class RestaurantProcessor(DataProcessor):
    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        if len(self.labels) == 3:
            return ['positive', 'neutral', 'negative']
        elif len(self.labels) == 4:
            return ['positive', 'neutral', 'negative', 'conflict']
        else:
            return list(self.labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        j = 0
        for i in range(0, len(lines), 3):
            guid = "%s-%s" % (set_type, j)
            j += 1
            text_left, _, text_right = [s.lower().strip() for s in lines[i][0].partition("$T$")]
            aspect = lines[i + 1][0].lower().strip()
            text_a = text_left + " " + aspect + " " + text_right  # 句子
            text_b = "What do you think of the " + aspect + " of it ?"
            # text_b = aspect
            label = lines[i + 2][0].strip()  # 标签
            self.labels.add(label)
            text_without_aspect = text_left + " " + text_right
            text_left_with_aspect = text_left + " " + aspect
            text_right_with_aspect = aspect + " " + text_right  # 注意没有倒序

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, aspect=aspect,
                             text_without_target=text_without_aspect,
                             text_left_with_target=text_left_with_aspect,
                             text_right_with_target=text_right_with_aspect,
                             text_left=text_left))
        return examples


class LaptopProcessor(DataProcessor):
    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        if len(self.labels) == 3:
            return ['positive', 'neutral', 'negative']
        elif len(self.labels) == 4:
            return ['positive', 'neutral', 'negative', 'conflict']
        else:
            return list(self.labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        j = 0
        # if set_type == 'train':
        #     del_list = np.random.choice(range(0, len(lines), 3), 600, replace=False)  # 不重复取
        for i in range(0, len(lines), 3):
            # if set_type == 'train' and i in del_list:
            #     continue
            guid = "%s-%s" % (set_type, j)
            j += 1
            text_left, _, text_right = [s.lower().strip() for s in lines[i][0].partition("$T$")]
            aspect = lines[i + 1][0].lower().strip()
            text_a = text_left + " " + aspect + " " + text_right  # 句子
            text_b = "What do you think of the " + aspect + " of it ?"
            label = lines[i + 2][0].strip()  # 标签
            self.labels.add(label)
            text_without_aspect = text_left + " " + text_right
            text_left_with_aspect = text_left + " " + aspect
            text_right_with_aspect = aspect + " " + text_right  # 注意没有倒序

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, aspect=aspect,
                             text_without_target=text_without_aspect,
                             text_left_with_target=text_left_with_aspect,
                             text_right_with_target=text_right_with_aspect,
                             text_left=text_left))
        return examples


class TweetProcessor(DataProcessor):
    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        if len(self.labels) == 3:
            return ['1', '0', '-1']
        else:
            return list(self.labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        j = 0
        for i in range(0, len(lines), 3):
            guid = "%s-%s" % (set_type, j)
            j += 1
            text_left, _, text_right = [s.lower().strip() for s in lines[i][0].partition("$T$")]
            aspect = lines[i + 1][0].lower().strip()
            text_a = text_left + " " + aspect + " " + text_right  # 句子
            text_b = "What do you think of the " + aspect + " of it ?"
            label = lines[i + 2][0].strip()  # 标签
            self.labels.add(label)
            text_without_aspect = text_left + " " + text_right
            text_left_with_aspect = text_left + " " + aspect
            text_right_with_aspect = aspect + " " + text_right  # 注意没有倒序

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, aspect=aspect,
                             text_without_target=text_without_aspect,
                             text_left_with_target=text_left_with_aspect,
                             text_right_with_target=text_right_with_aspect,
                             text_left=text_left))
        return examples

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 input_t_ids, input_t_mask, segment_t_ids,
                 input_without_t_ids, input_without_t_mask, segment_without_t_ids,
                 input_left_t_ids, input_left_t_mask, segment_left_t_ids,
                 input_right_t_ids, input_right_t_mask, segment_right_t_ids,
                 input_left_ids, input_left_mask, segment_left_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_t_ids = input_t_ids
        self.input_t_mask = input_t_mask
        self.segment_t_ids = segment_t_ids
        self.input_without_t_ids = input_without_t_ids
        self.input_without_t_mask = input_without_t_mask
        self.segment_without_t_ids = segment_without_t_ids
        self.input_left_t_ids = input_left_t_ids
        self.input_left_t_mask = input_left_t_mask
        self.segment_left_t_ids = segment_left_t_ids
        self.input_right_t_ids = input_right_t_ids
        self.input_right_t_mask = input_right_t_mask
        self.segment_right_t_ids = segment_right_t_ids
        self.input_left_ids = input_left_ids
        self.input_left_mask = input_left_mask
        self.segment_left_ids = segment_left_ids


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, aspect=None, text_without_target=None,
                 text_left_with_target=None, text_right_with_target=None, text_left=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.aspect = aspect  # add by gzj
        self.text_without_target = text_without_target  # add by gzj
        self.text_left_with_target = text_left_with_target  # add by gzj
        self.text_right_with_target = text_right_with_target  # add by gzj
        self.text_left = text_left