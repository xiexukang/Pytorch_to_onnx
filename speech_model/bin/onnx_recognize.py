# Copyright 2020 Mobvoi Inc. All Rights Reserved.
# Author: lyguo@mobvoi.com (Liyong Guo)

from __future__ import print_function
import argparse
import copy
import logging
import os
import sys

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import onnxruntime

from speech_model.dataset.dataset import CollateFunc, AudioDataset
from speech_model.transformer.encoder import TransformerEncoder
from speech_model.transformer.encoder import ConformerEncoder
from speech_model.transformer.decoder import TransformerDecoder
from speech_model.transformer.ctc import CTC
from speech_model.transformer.asr_model import ASRModel
from speech_model.utils.checkpoint import load_checkpoint
from speech_model.utils.mask import subsequent_mask
from speech_model.utils.mask import mask_finished_scores
from speech_model.utils.mask import mask_finished_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    # parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    # parser.add_argument('--penalty',
    #                     type=float,
    #                     default=0.0,
    #                     help='length penalty')
    parser.add_argument('--onnx_encoder_path',
                        required=True,
                        help='encoder model')
    parser.add_argument('--onnx_decoder_init_path',
                        required=True,
                        help='decoder init model')
    parser.add_argument('--onnx_decoder_non_init_path',
                        required=True,
                        help='decoder non-init model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    # parser.add_argument('--mode',
    #                     choices=[
    #                         'attention', 'ctc_greedy_search',
    #                         'ctc_prefix_beam_search', 'attention_rescoring'
    #                     ],
    #                     default='attention',
    #                     help='decoding mode')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
    #                  ] and args.batch_size > 1:
    #     logging.fatal(
    #         'decoding mode {} must be running with batch_size == 1'.format(
    #             args.mode))
    #     sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin)

    # Init dataset and data loader
    test_collate_conf = copy.copy(configs['collate_conf'])
    test_collate_conf['spec_aug'] = False
    test_collate_func = CollateFunc(**test_collate_conf, cmvn=args.cmvn)
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['sort'] = False
    test_dataset = AudioDataset(args.test_data, **dataset_conf)
    test_data_loader = DataLoader(test_dataset,
                                  collate_fn=test_collate_func,
                                  shuffle=False,
                                  batch_size=1,
                                  num_workers=0)

    input_dim = test_dataset.input_dim
    vocab_size = test_dataset.output_dim

    # Load dict
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1
    sos = eos
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # with torch.no_grad(), open(args.result_file, 'w') as fout:
    for batch_idx, batch in enumerate(test_data_loader):
        keys, feats, target, feats_lengths, target_lengths = batch
        feats = feats.to(device)
        target = target.to(device)
        feats_lengths = feats_lengths.to(device)
        target_lengths = target_lengths.to(device)
        ort_session = onnxruntime.InferenceSession(args.onnx_encoder_path)
        ort_feats = feats.detach().numpy()
        ort_lengths = feats_lengths.detach().numpy()

        ort_inputs = {
            ort_session.get_inputs()[0].name: ort_feats,
            ort_session.get_inputs()[1].name: ort_lengths
        }
        ort_outs = ort_session.run(None, ort_inputs)
        enc = ort_outs[0]
        enc_mask = ort_outs[1]
        encoder_out = torch.from_numpy(enc)
        encoder_mask = torch.from_numpy(enc_mask == 1)

        # cache: Optional[List[torch.Tensor]] = None
        batch_size = 1
        # beam_size = args.beam_size
        beam_size = 1
        running_size = batch_size * beam_size
        maxlen = 10
        cache = None
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  #(B*N, 1, max_len)

        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(sos)  # (B*N, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        for i in range(1, maxlen + 1):
            if end_flag.sum() == running_size:
                break
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)

            if not cache:
                ort_session = onnxruntime.InferenceSession(
                    args.onnx_decoder_init_path)
                ort_inputs = {
                    ort_session.get_inputs()[0].name: encoder_out.numpy(),
                    ort_session.get_inputs()[1].name: encoder_mask.numpy(),
                    ort_session.get_inputs()[2].name: hyps.numpy(),
                    ort_session.get_inputs()[3].name: hyps_mask.numpy()
                }
                ort_outs = ort_session.run(None, ort_inputs)
                logp = torch.from_numpy(ort_outs[0])
                cache = [torch.from_numpy(e) for e in ort_outs[1:]]
            elif cache:
                ort_session = onnxruntime.InferenceSession(
                    args.onnx_decoder_non_init_path)
                cache_np = [e.numpy() for e in cache]
                ort_inputs = {
                    ort_session.get_inputs()[0].name: encoder_out.numpy(),
                    ort_session.get_inputs()[1].name: encoder_mask.numpy(),
                    ort_session.get_inputs()[2].name: hyps.numpy(),
                    ort_session.get_inputs()[3].name: hyps_mask.numpy(),
                    ort_session.get_inputs()[4].name: cache_np[0],
                    ort_session.get_inputs()[5].name: cache_np[1],
                    ort_session.get_inputs()[6].name: cache_np[2],
                    ort_session.get_inputs()[7].name: cache_np[3],
                    ort_session.get_inputs()[8].name: cache_np[4],
                    ort_session.get_inputs()[9].name: cache_np[5]
                }
                ort_outs = ort_session.run(None, ort_inputs)
                logp = torch.from_numpy(ort_outs[0])
                cache = [torch.from_numpy(e) for e in ort_outs[1:]]

            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, eos)
            # 2.3 Seconde beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  #(B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  #(B*N)
            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  #(B*N)

            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  #(B*N, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  #(B*N, i+1)
            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_index = torch.argmax(scores, dim=-1).long()
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        content = ''
        for w in best_hyps[0]:
            if w == eos: break
            content += char_dict[int(w)]
        logging.info('{}'.format(content))
