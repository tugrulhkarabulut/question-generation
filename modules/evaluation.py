import argparse
import pickle

from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

def blue(hyps, refs):
    refs = [[r] for r in refs]
    score = corpus_bleu(refs, hyps)
    return score

def rouge(hyps, refs):
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']

def get_refs_and_hyps(hyp_path, ref_path):
    with open(hyp_path, 'rb') as f:
        hyps = pickle.load(f)

    with open(ref_path, 'rb') as f:
        refs = pickle.load(f)

    return hyps, refs

def save_metrics(metrics, path):
    with open(path, 'w') as f:
        f.write(','.join(metrics.keys()))
        f.write(','.join(metrics.values()))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculates evaluation metrics for the given reference and generation questions.')
    parser.add_argument('--ref_input', type=str, help='Path for the reference input data', default='./squad_val.pkl')
    parser.add_argument('--hyp_input', type=str, help='Path for the hypothesis input data', default='./squad_val_generated.pkl')
    parser.add_argument('--out', type=str, help='Ouput path for the metrics', default='./metrics.txt')
    parser.add_argument('--metric', type=str, help='Evaulation metric', choices=['blue', 'rouge', 'both'], default='both')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()

    hyps, refs = get_refs_and_hyps(args.hyp_input, args.ref_input)

    metrics = {}

    if args.metric == 'both':
        blue_score = blue(hyps, refs)
        metrics['bleu'] = blue_score
        rouge_scores = rouge(hyps, refs)
        metrics['rouge-1'] = rouge_scores[0]
        metrics['rouge-2'] = rouge_scores[1]
        metrics['rouge-l'] = rouge_scores[2]
    elif args.metric == 'bleu':
        blue_score = blue(hyps, refs)
        metrics['bleu'] = blue_score
    elif args.metric == 'rouge':
        rouge_scores = rouge(hyps, refs)
        metrics['rouge-1'] = rouge_scores[0]
        metrics['rouge-2'] = rouge_scores[1]
        metrics['rouge-l'] = rouge_scores[2]

    save_metrics(metrics, args.out)

    


    