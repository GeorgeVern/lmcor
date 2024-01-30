import argparse
import evaluate
from train_t5 import _create_corrector_data
from t5_utils import MODELS_DIR
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from io_utils import cont_write_textfile, read_textfile

DATA_PATH = 'data/{}/'

task_features = {'e2e_nlg_cleaned': ('meaning_representation', 'human_reference'), 'xsum': ('document', 'summary'),
                 'wmt22': ('source', 'reference')}

MODEL = "google/t5-v1_1-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL, legacy=False)

# MT-specific parameters
SRC = 'en'
TGT = 'de'

# corrector parameters
FILE_SAMPLE = '{}/{}_xglm-2.9b_sample-t0.6_4_5-shot.txt'
FILE_GREEDY = '{}/{}_xglm-2.9b_greedy_5-shot.txt'

NUM_SAMPLES = 4

# the following 2 hyperparameters are task-specific
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 256

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main(args):
    print("---task: {}".format(args.task))
    data_dir = DATA_PATH.format(args.task)

    if args.ckpt:
        model_path = args.ckpt
    else:
        model_path = MODEL

    print("---model: {}".format(model_path))

    source_data, references = [], []
    if args.task == 'wmt22':
        data_dir += '/{}-{}/'.format(SRC, TGT)
        source_data = read_textfile('{}/{}.{}'.format(data_dir, args.split, SRC))
        references = read_textfile('{}/{}.{}'.format(data_dir, args.split, TGT))
    else:
        task_data = load_dataset(args.task, split=args.split)
        for datapoint in task_data:
            source_data.append(datapoint[task_features[args.task][0]])
            references.append(datapoint[task_features[args.task][1]])

    if args.corrector:
        sampled_data_file = FILE_SAMPLE.format(data_dir, args.split)
        greedy_data_file = FILE_GREEDY.format(data_dir, args.split)
        single_cand = True if args.single_candidate in model_path else False
        source_data = _create_corrector_data(source_data, sampled_data_file, greedy_data_file,
                                             single_cand)

    model = T5ForConditionalGeneration.from_pretrained(model_path, cache_dir=MODELS_DIR)
    model = model.to(device)

    model_outputs = []

    for i in tqdm(range(0, len(source_data), args.bsize)):
        source_batch = source_data[i:i + args.bsize]
        inputs = tokenizer(source_batch, padding="longest", return_tensors="pt", max_length=MAX_SOURCE_LENGTH,
                           truncation=True).to(device)

        num_return_sequences = NUM_SAMPLES + 1 if args.sample else 1

        if args.sample:
            with torch.no_grad():
                gen_tokens = model.generate(**inputs, do_sample=True, max_new_tokens=MAX_TARGET_LENGTH, temperature=0.8,
                                            num_return_sequences=num_return_sequences)
        else:
            with torch.no_grad():
                gen_tokens = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_TARGET_LENGTH, num_beams=5)

        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        answers = [text.strip() for i, text in enumerate(gen_text)]
        if args.ckpt:
            cont_write_textfile(answers, model_path + "/model_preds.txt")
        else:
            cont_write_textfile(answers, FILE_GREEDY.format(args.task, args.split))
        model_outputs += answers

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=model_outputs, references=references)
    print(results)

    if args.task != 'wmt22':
        rouge = evaluate.load("rouge")
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

    main(args)
