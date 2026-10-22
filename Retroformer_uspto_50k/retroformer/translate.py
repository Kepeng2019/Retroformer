from utils.smiles_utils import *
from utils.translate_utils import translate_batch_original, translate_batch_stepwise
from utils.build_utils import build_model, build_iterator, load_checkpoint

import re
import os
import copy
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device GPU/CPU')
parser.add_argument('--batch_size_val', type=int, default=4, help='batch size')
parser.add_argument('--batch_size_trn', type=int, default=4, help='batch size')
parser.add_argument('--beam_size', type=int, default=10, help='beam size')
parser.add_argument('--stepwise', type=str, default='False', choices=['True', 'False'], help='')
parser.add_argument('--use_template', type=str, default='False', choices=['True', 'False'], help='')

parser.add_argument('--encoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--decoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
parser.add_argument('--d_ff', type=int, default=2048, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--known_class', type=str, default='True', help='with reaction class known/unknown')
parser.add_argument('--shared_vocab', type=str, default=False, choices=['True', 'False'], help='whether sharing vocab')
parser.add_argument('--shared_encoder', type=str, default=False, choices=['True', 'False'],
                    help='whether sharing encoder')

parser.add_argument('--data_dir', type=str, default='./data/template', help='base directory')
parser.add_argument('--intermediate_dir', type=str, default='./intermediate', help='intermediate directory')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='checkpoint directory')
parser.add_argument('--checkpoint', type=str, help='checkpoint model file')
parser.add_argument('--mode', default='test', type=str, help='choose the test file')

args = parser.parse_args()



# args.batch_size_val = 1
# args.shared_vocab = 'False'
# args.shared_encoder = False
# args.data_dir = 'data'
# args.intermediate_dir = './intermediate'
# args.checkpoint_dir = 'checkpoint'
# args.checkpoint = 'model_550000.pt'
# args.known_class = 'False'
# args.beam_size = 1
# args.stepwise = 'False'
# args.use_template = False


def translate(iterator, model, dataset):
    ground_truths = []
    generations = []
    classes = []
    if args.shared_vocab == True:
        invalid_token_indices = [dataset.tgt_stoi['<RX_{}>'.format(i)] for i in range(0, 113)]
        invalid_token_indices += [dataset.tgt_stoi['<UNK>'], dataset.tgt_stoi['<unk>']]
    else:
        invalid_token_indices = [dataset.tgt_stoi['<unk>']]
        # or invalid_token_indices = [dataset.tgt_stoi['<UNK>'], dataset.tgt_stoi['<unk>']], should check tgt_stoi
    # Translate:
    for batch in tqdm(iterator, total=len(iterator)):
        src, tgt, _, _, _ = batch

        if args.stepwise == 'False':
            # Original Main:
            pred_tokens, pred_scores = translate_batch_original(model, batch, beam_size=args.beam_size,
                                                                invalid_token_indices=invalid_token_indices)
            for idx in range(batch[0].shape[1]):
                gt = ''.join(dataset.reconstruct_smi(tgt[:, idx], src=False))
                hypos = np.array([''.join(dataset.reconstruct_smi(tokens, src=False)) for tokens in pred_tokens[idx]])
                hypo_len = np.array([len(smi_tokenizer(ht)) for ht in hypos])
                new_pred_score = copy.deepcopy(pred_scores[idx]).cpu().numpy() / hypo_len
                ordering = np.argsort(new_pred_score)[::-1]

                ground_truths.append(gt)
                generations.append(hypos[ordering])
                classes.append(dataset.src_itos[src[0,idx].cpu().detach().numpy()])
        else:
            # Stepwise Main:
            # untyped: T=10; beta=0.5, percent_aa=40, percent_ab=40
            # typed: T=10; beta=0.5, percent_aa=40, percent_ab=55
            print('do not use this parameters')
            if args.known_class == 'True':
                percent_ab = 55
            else:
                percent_ab = 40
            pred_tokens, pred_scores, predicts = \
                translate_batch_stepwise(model, batch, beam_size=args.beam_size,
                                         invalid_token_indices=invalid_token_indices,
                                         T=10, alpha_atom=-1, alpha_bond=-1,
                                         beta=0.5, percent_aa=40, percent_ab=percent_ab, k=3,
                                         use_template=args.use_template == 'True',
                                         factor_func=dataset.factor_func,
                                         reconstruct_func=dataset.reconstruct_smi,
                                         rc_path=args.intermediate_dir + '/rt2reaction_center.pk')

            original_beam_size = pred_tokens.shape[1]
            current_i = 0
            for batch_i, predict in enumerate(predicts):
                gt = ''.join(dataset.reconstruct_smi(tgt[:, batch_i], src=False))
                remain = original_beam_size
                beam_size = math.ceil(original_beam_size / len(predict))

                # normalized_reaction_center_score = np.array([pred[1] for pred in predict]) / 10
                hypo_i, hypo_scores_i = [], []
                for j, (rc, rc_score) in enumerate(predict):
                    # rc_score = normalized_reaction_center_score[j]

                    pred_token = pred_tokens[current_i + j]

                    sub_hypo_candidates, sub_score_candidates = [], []
                    for k in range(pred_token.shape[0]):
                        hypo_smiles_k = ''.join(dataset.reconstruct_smi(pred_token[k], src=False))
                        hypo_lens_k = len(smi_tokenizer(hypo_smiles_k))
                        hypo_scores_k = pred_scores[current_i + j][k].cpu().numpy() / hypo_lens_k + rc_score / 10

                        if hypo_smiles_k not in hypo_i:  # only select unique entries
                            sub_hypo_candidates.append(hypo_smiles_k)
                            sub_score_candidates.append(hypo_scores_k)

                    ordering = np.argsort(sub_score_candidates)[::-1]
                    sub_hypo_candidates = list(np.array(sub_hypo_candidates)[ordering])[:min(beam_size, remain)]
                    sub_score_candidates = list(np.array(sub_score_candidates)[ordering])[:min(beam_size, remain)]

                    hypo_i += sub_hypo_candidates
                    hypo_scores_i += sub_score_candidates

                    remain -= beam_size

                current_i += len(predict)
                ordering = np.argsort(hypo_scores_i)[::-1][:args.beam_size]
                ground_truths.append(gt)
                generations.append(np.array(hypo_i)[ordering])
                classes.append(dataset.src_itos[src[0,idx].cpu().detach().numpy()])


    return ground_truths, generations,classes


def main(args):
    # Build Data Iterator:
    iterator, dataset = build_iterator(args, train=False)

    # Load Checkpoint Model:
    model = build_model(args, dataset.src_itos, dataset.tgt_itos)
    _, _, model = load_checkpoint(args, model)

   

    # Begin Translating:
    ground_truths, generations,classes = translate(iterator, model, dataset)

    # Save Results:
    file_name1 = 'ground_truth_beam_{}_{}_{}_known_class_{}.csv'.format(args.beam_size,args.checkpoint,args.mode,args.known_class)
    file_name2 = 'generations_beam_{}_{}_{}_known_class_{}.csv'.format(args.beam_size,args.checkpoint,args.mode,args.known_class)
    output_path1 = os.path.join('result/',file_name1)
    output_path2 = os.path.join('result/',file_name2)
    os.makedirs('result', exist_ok=True)

    pd.concat((pd.DataFrame(classes),pd.DataFrame(ground_truths)),axis=1).to_csv(output_path1,index=False)
    pd.DataFrame(generations).to_csv(output_path2,index=None)
    

    accuracy_matrix = np.zeros((len(ground_truths), args.beam_size))
    for i in range(len(ground_truths)):
        gt_i = canonical_smiles(ground_truths[i])
        generation_i = [canonical_smiles(gen) for gen in generations[i]]
        for j in range(args.beam_size):
            if gt_i in generation_i[:j + 1]:
                accuracy_matrix[i][j] = 1

    # with open(output_path, 'wb') as f:
    #     pickle.dump((ground_truths, generations), f)

    for j in range(args.beam_size):
        print('Top-{}: {}'.format(j + 1, round(np.mean(accuracy_matrix[:, j]), 4)))

    return


if __name__ == "__main__":
    print(args)
    if args.known_class == 'True':
        args.checkpoint_dir = args.checkpoint_dir + '_typed'
    else:
        args.checkpoint_dir = args.checkpoint_dir + '_untyped'
    if args.use_template == 'True':
        args.stepwise = 'True'
    main(args)
