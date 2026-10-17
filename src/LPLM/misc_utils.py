import numpy as np
import torch
from unidecode import unidecode
import src.util as ut
import csv
from src.LPLM.extension import *
from tqdm import tqdm
import subprocess
from torch.utils.data import Dataset

# This function returns geometric mean of a list


def g_mean(list_):
    log_list_ = np.log(list_)
    return np.exp(log_list_.mean())


# This function transforms the LIKE patterns to new language
def LIKE_pattern_to_newLanguage(pattern_list, pattern_type):
    transformed_pattern = ''
    for pattern in pattern_list:
        if len(pattern) == 1:
            transformed_pattern += pattern
        else:
            new_pattern = ''
            count = 0
            for char in pattern:
                if count < 1:
                    new_pattern += char
                    count += 1
                else:
                    if (
                        new_pattern[-1] not in ('_', '@')
                        and char not in ('_', '@')
                    ):
                        new_pattern += char + '$'
                    else:
                        new_pattern += char
            transformed_pattern += new_pattern
    if pattern_type == 'prefix':
        transformed_pattern = f'{transformed_pattern[0]}.{transformed_pattern[1:]}'
    elif pattern_type == 'suffix':
        transformed_pattern += '#'
    elif pattern_type == 'end_underscore':
        transformed_pattern = (
            f'{transformed_pattern[0]}.{transformed_pattern[1:]}')
    elif pattern_type == 'begin_underscore':
        transformed_pattern = (
            f'{transformed_pattern[0]}{transformed_pattern[1:]}#')
    elif pattern_type == 'prefix_suffix':
        transformed_pattern = (
            f'{transformed_pattern[0]}.{transformed_pattern[1:]}#')
    return transformed_pattern


# This function computes loss
def binary_crossentropy(preds, targets, mask):
    loss = targets * torch.log(preds + 0.00001) + \
        (1 - targets) * torch.log(1 - (preds - 0.00001))
    if mask is not None:
        loss = mask * loss
    return - torch.sum(loss) / torch.sum(mask)


def name2numpy(name, char2idx):
    tensor = np.zeros((len(name), len(char2idx)), dtype=np.int8)
    for i, char in enumerate(name):
        if char not in char2idx:
            raise ValueError(
                f"LPLM vocabulary cannot encode character U+{ord(char):04X}"
            )
        tensor[i][char2idx[char]] = 1
    return tensor


def convertQueries2testData(queries, char2idx):
    inputs = []
    length = []
    for query in queries:
        transformedLikepattern, is_end_esc = LIKE_pattern_to_extendedLanguage(
            query)
        transformed_to_tensor = name2numpy(transformedLikepattern, char2idx)
        inputs.append(transformed_to_tensor)
        length.append(len(transformed_to_tensor))
    maxs = max(length)

    padded_inputs = []
    masks = []
    for np_mask in inputs:
        old_len = len(np_mask)
        np_mask = np.concatenate(
            (np_mask, np.zeros((maxs - len(np_mask), len(char2idx)), dtype=np.int8)), 0)
        padded_inputs.append(np_mask)
        masks.append([1] * old_len + [0] * (maxs - old_len))
    test_dataset = [(torch.FloatTensor(padded_inputs[i]), torch.tensor(masks[i]))
                    for i in range(len(masks))]
    return test_dataset

# This function loads LIKE-patterns with ground truth probabilities


def loadtrainData(filename, char2idx):
    inputs = []
    targets = []
    length = []
    n_data = int(subprocess.run(
        ['wc', '-l', filename], capture_output=True, text=True).stdout.split()[0])
    print(f'Load data from "{filename}"')
    with open(filename, newline='', encoding='utf-8') as file:
        for line_ in tqdm(csv.reader(file), total=n_data):
            assert len(line_) == 2, line_
            transformedLikepattern, is_end_esc = LIKE_pattern_to_extendedLanguage(
                line_[0])
            transformed_to_tensor = name2numpy(
                transformedLikepattern, char2idx)
            if transformed_to_tensor is None:
                continue
            inputs.append(transformed_to_tensor)
            length.append(len(transformed_to_tensor))
            ground_prob_list = [
                float(element) for element in line_[-1].split(' ')]
            targets.append(ground_prob_list)
    return inputs, targets, max(length)

# This function pads the zero vectors to like-patterns.


class CompactTrainDataset(Dataset):
    """Store token ids and targets; create one-hot batches on the training device."""

    def __init__(self, filename, char2idx):
        self.alphabet_size = len(char2idx)
        n_data = int(subprocess.run(
            ['wc', '-l', filename], capture_output=True, text=True).stdout.split()[0])

        print(f'Scan compact training data from "{filename}"')
        n_valid = 0
        max_length = 0
        with open(filename, newline='', encoding='utf-8') as file:
            for line_ in tqdm(csv.reader(file), total=n_data):
                assert len(line_) == 2, line_
                transformed, _ = LIKE_pattern_to_extendedLanguage(line_[0])
                missing = {char for char in transformed if char not in char2idx}
                if missing:
                    points = ", ".join(f"U+{ord(char):04X}" for char in sorted(missing))
                    raise ValueError(
                        f'LPLM vocabulary cannot encode {points} in "{filename}"'
                    )
                target_length = len(line_[-1].split(' '))
                max_length = max(max_length, len(transformed), target_length)
                n_valid += 1

        token_dtype = np.uint8 if self.alphabet_size <= 256 else np.uint16
        self.token_ids = np.zeros((n_valid, max_length), dtype=token_dtype)
        self.lengths = np.empty(n_valid, dtype=np.int32)
        self.targets = np.zeros((n_valid, max_length), dtype=np.float32)

        print(f'Build compact training data from "{filename}"')
        row_index = 0
        with open(filename, newline='', encoding='utf-8') as file:
            for line_ in tqdm(csv.reader(file), total=n_data):
                transformed, _ = LIKE_pattern_to_extendedLanguage(line_[0])
                missing = {char for char in transformed if char not in char2idx}
                if missing:
                    points = ", ".join(f"U+{ord(char):04X}" for char in sorted(missing))
                    raise ValueError(
                        f'LPLM vocabulary cannot encode {points} in "{filename}"'
                    )

                token_ids = [char2idx[char] for char in transformed]
                target = [float(element) for element in line_[-1].split(' ')]
                self.token_ids[row_index, :len(token_ids)] = token_ids
                self.lengths[row_index] = len(token_ids)
                self.targets[row_index, :len(target)] = target
                row_index += 1

        size_gib = (
            self.token_ids.nbytes + self.lengths.nbytes + self.targets.nbytes
        ) / (1024 ** 3)
        print(
            f'Compact training dataset: rows={n_valid}, max_length={max_length}, '
            f'memory={size_gib:.2f} GiB'
        )

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return self.token_ids[index], self.lengths[index], self.targets[index]


def addpaddingTrain(filename, char2idx):
    return CompactTrainDataset(filename, char2idx)


# This function takes a file path that contains test LIKE-patterns
def loadtestData(filename, char2idx):
    inputs = []
    length = []
    actual_card = []
    queries = []
    n_data = int(subprocess.run(
        ['wc', '-l', filename], capture_output=True, text=True).stdout.split()[0])
    print(f'Load data from "{filename}"')
    with open(filename, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=n_data):
            line = line.rstrip('\n\r')
            if not line:
                continue
            pattern, card = line.rsplit(',', 1)
            transformedLikepattern, is_end_esc = LIKE_pattern_to_extendedLanguage(
                pattern)
            # transformedLikepattern = LIKE_pattern_to_newLanguage(line_[0].replace(' ', '@').split('%'), line_[1])
            # transformed_to_tensor = name2tensor(unidecode(transformedLikepattern), char2idx)
            transformed_to_tensor = name2numpy(
                transformedLikepattern, char2idx)
            inputs.append(transformed_to_tensor)
            length.append(len(transformed_to_tensor))
            actual_card.append(int(card))
            queries.append(pattern)
    if not inputs:
        raise ValueError(f'No usable LIKE patterns in "{filename}"')
    return inputs, max(length), actual_card, queries


def loadtestQueries(filename, char2idx):
    queries = []
    actual_cards = []
    n_data = int(subprocess.run(
        ['wc', '-l', filename], capture_output=True, text=True).stdout.split()[0])
    print(f'Load queries from "{filename}"')
    with open(filename, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=n_data):
            line = line.rstrip('\n\r')
            if not line:
                continue
            pattern, card = line.rsplit(',', 1)
            transformed, _ = LIKE_pattern_to_extendedLanguage(pattern)
            missing = {char for char in transformed if char not in char2idx}
            if missing:
                points = ", ".join(f"U+{ord(char):04X}" for char in sorted(missing))
                raise ValueError(
                    f'LPLM vocabulary cannot encode {points} in "{filename}"'
                )
            queries.append(pattern)
            actual_cards.append(int(card))
    if not queries:
        raise ValueError(f'No usable LIKE patterns in "{filename}"')
    return queries, actual_cards


def addpaddingTest(filename, char2idx):
    padded_inputs = []
    masks = []
    inputs, maxs, actual_card, test_queries = loadtestData(
        filename, char2idx)

    for np_mask in inputs:
        old_len = len(np_mask)
        np_mask = np.concatenate(
            (np_mask, np.zeros((maxs - len(np_mask), len(char2idx)), dtype=np.int8)), 0)
        padded_inputs.append(np_mask)
        masks.append([1] * old_len + [0] * (maxs - old_len))
    test_dataset = [(torch.FloatTensor(padded_inputs[i]), torch.tensor(
        masks[i]), torch.tensor(actual_card[i])) for i in range(len(masks))]
    return test_dataset, test_queries, actual_card


# This function compute and return q-error
def compute_qerrors(actual_card, estimated_card):
    return max(actual_card/estimated_card, estimated_card/actual_card)


def get_vocab(trainfile):

    vocab_dict = {}
    with open(trainfile, encoding='utf-8') as file:
        for i in file:
            i = i.strip().split(':')[0]
            for k in i:
                if k != '%' and k not in vocab_dict:
                    vocab_dict[k] = 0
    vocab = ''
    for token in vocab_dict:
        vocab += token
    return vocab + '$.#'


def get_vocab_from_list(lines, max_n_char=None):
    # Every structural wildcard token must be reserved. Treating these as
    # ordinary unknown characters changes LIKE semantics; omitting them used to
    # make the loaders silently drop queries containing long underscore runs.
    special_chars = ''.join(esc_list)
    db = [line.rstrip('\n') for line in lines]
    char_set = ut.char_set_from_db(db, True)

    for special_char in special_chars:
        if special_char in char_set:
            char_set.remove(special_char)

    if max_n_char is not None and len(char_set) > max_n_char - 1:
        char_set = char_set[:max_n_char]
        char_set[-1] = '언'

    print(''.join(sorted(char_set)) + special_chars)
    return ''.join(sorted(char_set)) + special_chars


# This function estimate the cardinalities and saves them to a txt file
def estimate_cardinality(test_dataset, model, device, save_file_path, dataset_size):
    # print(f"{save_file_path = }")
    # if save_file_path is not None:
    #     write_to_file = open(save_file_path, 'w')
    # qerror_list = []
    actual_cards = []
    estimated_cards = []
    with torch.no_grad():
        for name, mask, actual_card in test_dataset:
            bs = name.size()[0]
            name = name.to(device)
            output = model(name)
            output = output.to(device)
            if bs == 1:
                output = output.unsqueeze(dim=0)
            mask = mask.to(device)
            actual_cards.extend(actual_card.numpy())
            output = torch.prod(torch.pow(output, mask),
                                axis=-1) * dataset_size
            estimated_cards.extend(output.cpu().numpy())
            # qerrors = compute_qerrors(actual_card, output.item())
            # if save_file_path is not None:
            #     write_to_file.write(str(output.item()) + '\n')

    # print(f'G-mean: {np.round(g_mean(qerror_list), 4)}')
    # print(f'Mean: {np.round(np.average(qerror_list), 4)}')
    # print(f'Median: {np.round(np.percentile(qerror_list, 50), 4)}')
    # print(f'90th: {np.round(np.percentile(qerror_list, 90), 4)}')
    # print(f'99th: {np.round(np.percentile(qerror_list, 99), 4)}')
    return actual_cards, estimated_cards
