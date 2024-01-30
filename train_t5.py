import argparse
import os
from io_utils import read_textfile
from t5_utils import t5_trainer, MODELS_DIR
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from evaluate import load
import nltk

task_features = {'e2e_nlg_cleaned': ('meaning_representation', 'human_reference'), 'xsum': ('document', 'summary'),
                 'wmt22': ('source', 'reference')}

DATA_PATH = 'data/{}/'

# MT-specific parameters
SRC = 'en'
TGT = 'de'
TASK = None

MODEL = "google/t5-v1_1-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL, legacy=False)

# corrector parameters
FILE_SAMPLE = '/{}_polylm_sample-t0.6_4.txt'
FILE_GREEDY = '/{}_polylm_greedy_1.txt'

# the following 2 hyperparameters are task-specific
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 256
SELECTION_METRIC = 'bleu'

NUM_SAMPLES = 4
SENTINEL_TOKEN = " <extra_id_0> "
STEPS = 200
MAX_STEPS = 10000


def _create_corrector_data(input_data, sampled_data_file, greedy_data_file, single_cand=False,
                           sent_token=SENTINEL_TOKEN):
    """
    creates the input for the corrector model (greedy decoded & 4 sampled candidates)
    :param input_data: the input data
    :param sampled_data_file: the data sampled by the LLM
    :param greedy_data_file: the greedy output of the LLM
    :param single_cand: whether the corrector receives one or more candidates as input
    :param sent_token: the sentinel token
    :return:
    """
    sampled_candidates = read_textfile(sampled_data_file, NUM_SAMPLES)
    greedy_candidates = read_textfile(greedy_data_file, 0)

    corrector_input = []
    for i, example in enumerate(input_data):
        if single_cand:
            candidates = [greedy_candidates[i]]
        else:
            candidates = [greedy_candidates[i]] + sampled_candidates[i]
        corrector_input.append("source: {}{}candidates: ".format(example, sent_token) + (sent_token).join(candidates))

    return corrector_input


def _data2dict(input_data, target_data):
    dict_dataset = []
    for i, example in enumerate(input_data):
        dict_dataset.append({task_features[TASK][0]: example, task_features[TASK][1]: target_data[i]})
    return dict_dataset


def preprocess_dataset(examples):
    # encode the code-docstring pairs
    inputs = examples[task_features[TASK][0]]
    target_sents = examples[task_features[TASK][1]]

    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, padding="max_length", truncation=True)

    # encode the summaries
    labels = tokenizer(target_sents, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs


def _round_results(metric_result):
    if type(metric_result) != list:
        return round(metric_result, 4)
    else:
        return [round(x, 4) for x in metric_result]


def compute_metrics(eval_pred):
    metric = load(SELECTION_METRIC)

    predictions, labels = eval_pred

    # Replace -100 in the predictions as we can't decode them.
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # Extract a few results
    if SELECTION_METRIC == 'rouge':
        result = {key: value * 100 for key, value in result.items()}
    elif SELECTION_METRIC == 'bleu':
        result['bleu'] = result['bleu'] * 100
        result['precisions'] = [x * 100 for x in result['precisions']]

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: _round_results(v) for k, v in result.items()}


def main(args):
    global TASK  # Reference the global TASK variable
    TASK = args.task  # Set the value of TASK

    data_dir = DATA_PATH.format(TASK)
    os.makedirs(data_dir, exist_ok=True)

    if TASK != 'wmt22':
        training_data = load_dataset(TASK)
    else:
        data_dir += '/{}-{}/'.format(SRC, TGT)
        os.makedirs(data_dir, exist_ok=True)
        training_data_split = {}
        for split in ["train", 'validation']:
            source_data = read_textfile(data_dir + '/{}.{}'.format(split, SRC))
            references = read_textfile(data_dir + '/{}.{}'.format(split, TGT))
            training_data_split[split] = Dataset.from_list(_data2dict(source_data, references))
            training_data = DatasetDict(training_data_split)

    if args.corrector:
        training_data_split = {}
        for split in ["train", 'validation']:
            sampled_data_file = data_dir + FILE_SAMPLE.format(split)
            greedy_data_file = data_dir + FILE_GREEDY.format(split)
            corrector_input_train = _create_corrector_data(training_data[split][task_features[TASK][0]],
                                                           sampled_data_file,
                                                           greedy_data_file, args.single_candidate)

            training_data_split[split] = Dataset.from_list(
                _data2dict(corrector_input_train, training_data[split][task_features[TASK][1]]))

        training_data = DatasetDict(training_data_split)

    task_data_tokenized = training_data.map(preprocess_dataset, batched=True)

    t5_model = T5ForConditionalGeneration.from_pretrained(MODEL, cache_dir=MODELS_DIR)

    t5_trainer(output_dir=args.output_dir, model=t5_model, tokenizer=tokenizer,
               tokenized_dataset=task_data_tokenized,
               task_metrics=compute_metrics, batch_size=args.bsize, grad_acc_steps=args.grad_acc_steps, steps=STEPS,
               max_steps=MAX_STEPS,
               selection_metric=SELECTION_METRIC, max_target_length=MAX_TARGET_LENGTH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train corrector/baseline for a specific task.')
    parser.add_argument('--task', default='wmt22', choices=['e2e_nlg_cleaned', 'xsum', 'wmt22'], type=str,
                        help='the task name')
    parser.add_argument('--corrector', action='store_true', help='whether the T5 model is a corrector')
    parser.add_argument('--single_candidate', action='store_true',
                        help="whether the corrector receives one or more candidates as input")
    parser.add_argument('--bsize', default=8, type=int, help="the batch size")
    parser.add_argument('--grad_acc_steps', default=16, type=int, help="the gradient accumulation steps size")
    parser.add_argument('--output_dir', type=str, help="where to store the checkpoints")
    args = parser.parse_args()

    main(args)
