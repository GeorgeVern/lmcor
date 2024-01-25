import argparse
import evaluate
from datasets import load_dataset
from comet import download_model, load_from_checkpoint
from train_t5 import read_textfile
from sacrebleu import BLEU
from t5_utils import MODELS_DIR

DATA_PATH = 'data/{}/'

# MT-specific parameters
SRC = 'en'
TGT = 'de'

task_features = {'e2e_nlg_cleaned': ('meaning_representation', 'human_reference'), 'xsum': ('document', 'summary'),
                 'wmt22': ('source', 'reference')}


def main(args):
    predictions = read_textfile(args.hyp)

    split = 'test'
    if args.task != 'wmt22':
        test_data = load_dataset(args.task, split=split)
        references = [datapoint[task_features[args.task][1]] for datapoint in test_data]
    else:
        data_dir = DATA_PATH.format(args.task) + '/{}-{}/'.format(SRC, TGT)
        source_data = read_textfile('{}/{}.{}'.format(data_dir, split, SRC))
        references = read_textfile('{}/{}.{}'.format(data_dir, split, TGT))

    assert len(predictions) == len(references)

    bleu = BLEU()
    print(bleu.corpus_score(predictions, [references]))

    if args.task == 'e2e_nlg_cleaned':
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=predictions, references=references)
        print(results)
    elif args.task == 'wmt22':

        model_path = download_model("Unbabel/wmt22-cometkiwi-da", saving_directory=MODELS_DIR)
        model = load_from_checkpoint(model_path)
        data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(source_data, predictions, references)]
        model_output = model.predict(data, batch_size=200, gpus=1)
        print("---COMET score: ", model_output.system_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reproduce document-level metric scores from the paper.')
    parser.add_argument('--task', default="wmt22", type=str, help='the task name')
    parser.add_argument('--hyp', type=str, help='the hypothesis file')

    args = parser.parse_args()

    main(args)
