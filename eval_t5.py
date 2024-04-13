import argparse
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
from evaluate import load
from io_utils import cont_write_textfile, read_textfile
from t5_utils import MODELS_DIR
from train_t5 import _create_corrector_data

DATA_PATH = 'data/{}/'
task_features = {'e2e_nlg_cleaned': ('meaning_representation', 'human_reference'),
                 'xsum': ('document', 'summary'),
                 'wmt22': ('source', 'reference')}

MODEL = "google/t5-v1_1-base"
SRC = 'en'
TGT = 'de'
FILE_SAMPLE = '{}/{}_xglm-2.9b_sample-t0.6_4_5-shot.txt'
FILE_GREEDY = '{}/{}_xglm-2.9b_greedy_5-shot.txt'
NUM_SAMPLES = 4
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 256
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main(args):
    print(f"---task: {args.task}")
    data_dir = DATA_PATH.format(args.task)
    model_path = args.ckpt if args.ckpt else MODEL
    print(f"---model: {model_path}")

    source_data, references = load_data(args, data_dir)

    if args.corrector:
        source_data = prepare_corrector_data(args, data_dir, source_data)

    model = T5ForConditionalGeneration.from_pretrained(model_path, cache_dir=MODELS_DIR).to(device)
    model_outputs = generate_outputs(args, model, source_data)

    compute_and_print_metrics(args, model_outputs, references)


def load_data(args, data_dir):
    source_data, references = [], []

    if args.task == 'wmt22':
        data_dir += f'/{SRC}-{TGT}/'
        source_data = read_textfile(f'{data_dir}/{args.split}.{SRC}')
        references = read_textfile(f'{data_dir}/{args.split}.{TGT}')
    else:
        task_data = load_dataset(args.task, split=args.split)
        for datapoint in task_data:
            source_data.append(datapoint[task_features[args.task][0]])
            references.append(datapoint[task_features[args.task][1]])

    return source_data, references


def prepare_corrector_data(args, data_dir, source_data):
    sampled_data_file = FILE_SAMPLE.format(data_dir, args.split)
    greedy_data_file = FILE_GREEDY.format(data_dir, args.split)
    single_cand = args.single_candidate in args.model_path
    return _create_corrector_data(source_data, sampled_data_file, greedy_data_file, single_cand)


def generate_outputs(args, model, source_data):
    model_outputs = []

    for i in tqdm(range(0, len(source_data), args.bsize)):
        source_batch = source_data[i:i + args.bsize]
        inputs = tokenizer(source_batch, padding="longest", return_tensors="pt",
                           max_length=MAX_SOURCE_LENGTH, truncation=True).to(device)

        num_return_sequences = NUM_SAMPLES + 1 if args.sample else 1

        with torch.no_grad():
            gen_tokens = model.generate(**inputs, do_sample=args.sample, max_new_tokens=MAX_TARGET_LENGTH,
                                        temperature=0.8, num_return_sequences=num_return_sequences if args.sample else 1)

        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        answers = [text.strip() for text in gen_text]

        if args.ckpt:
            cont_write_textfile(answers, f"{args.model_path}/model_preds.txt")
        else:
            cont_write_textfile(answers, FILE_GREEDY.format(args.task, args.split))

        model_outputs += answers

    return model_outputs


def compute_and_print_metrics(args, model_outputs, references):
    bleu = load("bleu")
    results = bleu.compute(predictions=model_outputs, references=references)
    print(results)

    if args.task != 'wmt22':
        rouge = load("rouge")
        results = rouge.compute(predictions=model_outputs, references=references)
        print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the corrector/baseline for a specific task.')
    parser.add_argument('--task', type=str, choices=['e2e_nlg_cleaned', 'xsum', 'wmt22'], help='the task')
    parser.add_argument('--ckpt', help='the checkpoint file')
    parser.add_argument('--corrector', action='store_true', help='whether the T5 model is a corrector')
    parser.add_argument('--single_candidate', action='store_true',
                        help="whether the corrector receives one or more candidates as input")
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='test', help='the data split')
    parser.add_argument('--bsize', default=32, type=int, help="the batch size")
    parser.add_argument('--sample', action='store_true', help="whether to also sample from the model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, legacy=False)
    main(args)
