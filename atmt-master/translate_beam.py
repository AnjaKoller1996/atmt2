import os
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.serialization import default_restore_location

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.beam import BeamSearch, BeamSearchNode


def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', default=False, help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--data', default='data_asg4/prepared_data', help='path to data directory')
    parser.add_argument('--checkpoint-path', default='checkpoints_asg4/checkpoint_best.pt',
                        help='path to the model file')
    parser.add_argument('--batch-size', default=None, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--output', default='model_translations_nbest_k3_alpha09_gamma10neg.txt', type=str,
                        help='path to the output file destination')
    parser.add_argument('--max-len', default=100, type=int, help='maximum length of generated sequence')

    # Add beam search arguments
    parser.add_argument('--beam-size', default=3, type=int, help='number of hypotheses expanded in beam search')

    # Add normalization parameter alpha
    parser.add_argument('--alpha', default=1.0, type=float, help='length normalization parameter')

    # Add gamma parameter for diversity beam search
    parser.add_argument('--gamma', default=1.0, type=float, help='diversity parameter gamma')

    return parser.parse_args()


def main(args):
    """ Main translation function' """
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)  # sets the random seed from pytorch random number generators
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args_loaded = argparse.Namespace(**{**vars(args), **vars(state_dict['args'])})
    args_loaded.data = args.data
    args = args_loaded
    utils.init_logging(args)

    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.source_lang)))
    logging.info('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.target_lang)))
    logging.info('Loaded a target dictionary ({:s}) with {:d} words'.format(args.target_lang, len(tgt_dict)))

    # Load dataset
    test_dataset = Seq2SeqDataset(
        src_file=os.path.join(args.data, 'test.{:s}'.format(args.source_lang)),
        tgt_file=os.path.join(args.data, 'test.{:s}'.format(args.target_lang)),
        src_dict=src_dict, tgt_dict=tgt_dict)

    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=1, collate_fn=test_dataset.collater,
                                              batch_sampler=BatchSampler(test_dataset, 9999999,
                                                                         args.batch_size, 1, 0, shuffle=False,
                                                                         seed=args.seed))
    # Build model and criterion
    model = models.build_model(args, src_dict, tgt_dict)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {:s}'.format(args.checkpoint_path))
    progress_bar = tqdm(test_loader, desc='| Generation', leave=False)

    # Iterate over the test set
    all_hyps = {}
    for i, sample in enumerate(progress_bar):

        # Create a beam search object or every input sentence in batch
        batch_size = sample['src_tokens'].shape[0]  # returns number of rows from sample['src_tokens']
        searches = [BeamSearch(args.beam_size, args.max_len - 1, tgt_dict.unk_idx) for i in range(batch_size)]
        # beam search with beamsize, max seq length and unkindex --> do this B times

        with torch.no_grad():  # disables gradient calculation
            # Compute the encoder output
            encoder_out = model.encoder(sample['src_tokens'], sample['src_lengths'])
            # __QUESTION 1: What is "go_slice" used for and what do its dimensions represent?
            #  encoder_out = self.encoder(src_tokens, src_lengths) decoder_out = self.decoder(tgt_inputs, encoder_out)
            go_slice = \
                torch.ones(sample['src_tokens'].shape[0], 1).fill_(tgt_dict.eos_idx).type_as(sample['src_tokens'])
            # vector of ones of length sample['src_tokens'] rows and 1 col filled with eos_idx casted to type sample[
            # 'src_tokens']
            if args.cuda:
                go_slice = utils.move_to_cuda(go_slice)

            # Compute the decoder output at the first time step
            decoder_out, _ = model.decoder(go_slice, encoder_out)  # decoder out = decoder(tgt_inputs, encoder_out)

            # __QUESTION 2: Why do we keep one top candidate more than the beam size?
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)),
                                                    args.beam_size + 1, dim=-1)
            # returns largest k elements (here beam_size+1) of the input torch.log(torch.softmax(decoder_out,
            # dim=2) in dimension -1 + 1 is taken because the input is given in logarithmic notation

        #  Create number of beam_size beam search nodes for every input sentence
        for i in range(batch_size):
            for j in range(args.beam_size):
                best_candidate = next_candidates[i, :, j]
                backoff_candidate = next_candidates[i, :, j + 1]
                best_log_p = log_probs[i, :, j]
                backoff_log_p = log_probs[i, :, j + 1]
                next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
                log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                log_p = log_p[-1]

                # Store the encoder_out information for the current input sentence and beam
                emb = encoder_out['src_embeddings'][:, i, :]
                lstm_out = encoder_out['src_out'][0][:, i, :]
                final_hidden = encoder_out['src_out'][1][:, i, :]
                final_cell = encoder_out['src_out'][2][:, i, :]
                try:
                    mask = encoder_out['src_mask'][i, :]
                except TypeError:
                    mask = None

                node = BeamSearchNode(searches[i], emb, lstm_out, final_hidden, final_cell,
                                      mask, torch.cat((go_slice[i], next_word)), log_p, 1)

                # add normalization here according to paper
                lp = normalize(node.length)
                score = node.eval()/lp
                # Add diverse
                score = diverse(score, j)
                # __QUESTION 3: Why do we add the node with a negative score?
                searches[i].add(-score, node)

        # Start generating further tokens until max sentence length reached
        for _ in range(args.max_len - 1):

            # Get the current nodes to expand
            nodes = [n[1] for s in searches for n in s.get_current_beams()]
            if nodes == []:
                break  # All beams ended in EOS

            # Reconstruct prev_words, encoder_out from current beam search nodes
            prev_words = torch.stack([node.sequence for node in nodes])
            encoder_out["src_embeddings"] = torch.stack([node.emb for node in nodes], dim=1)
            lstm_out = torch.stack([node.lstm_out for node in nodes], dim=1)
            final_hidden = torch.stack([node.final_hidden for node in nodes], dim=1)
            final_cell = torch.stack([node.final_cell for node in nodes], dim=1)
            encoder_out["src_out"] = (lstm_out, final_hidden, final_cell)
            try:
                encoder_out["src_mask"] = torch.stack([node.mask for node in nodes], dim=0)
            except TypeError:
                encoder_out["src_mask"] = None

            with torch.no_grad():
                # Compute the decoder output by feeding it the decoded sentence prefix
                decoder_out, _ = model.decoder(prev_words, encoder_out)

            # see __QUESTION 2
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)), args.beam_size + 1,
                                                    dim=-1)

            #  Create number of beam_size next nodes for every current node
            for i in range(log_probs.shape[0]):
                for j in range(args.beam_size):

                    best_candidate = next_candidates[i, :, j]
                    backoff_candidate = next_candidates[i, :, j + 1]
                    best_log_p = log_probs[i, :, j]
                    backoff_log_p = log_probs[i, :, j + 1]
                    next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
                    log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                    log_p = log_p[-1]
                    next_word = torch.cat((prev_words[i][1:], next_word[-1:]))

                    # Get parent node and beam search object for corresponding sentence
                    node = nodes[i]
                    search = node.search

                    # __QUESTION 4: How are "add" and "add_final" different? What would happen if we did not make this distinction?

                    # Store the node as final if EOS is generated
                    if next_word[-1] == tgt_dict.eos_idx:
                        node = BeamSearchNode(search, node.emb, node.lstm_out, node.final_hidden,
                                              node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                                                                     next_word)), node.logp,
                                              node.length)
                        # Add length normalization
                        lp = normalize(node.length)
                        score = node.eval()/lp
                        # add diverse
                        score = diverse(score, j)
                        search.add_final(-score, node)

                    # Add the node to current nodes for next iteration
                    else:
                        node = BeamSearchNode(search, node.emb, node.lstm_out, node.final_hidden,
                                              node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                                                                     next_word)), node.logp + log_p,
                                              node.length + 1)
                        # Add length normalization
                        lp = normalize(node.length)
                        score = node.eval()/lp
                        # add diverse
                        score = diverse(score, j)
                        search.add(-score, node)

            # __QUESTION 5: What happens internally when we prune our beams?
            # How do we know we always maintain the best sequences?
            for search in searches:
                search.prune()

        # Segment into 1 best sentences
        #best_sents = torch.stack([search.get_best()[1].sequence[1:].cpu() for search in searches])

        # segment 3 best oneliner
        best_sents = torch.stack([n[1].sequence[1:] for s in searches for n in s.get_best()])

        # segment into n best sentences
        #for s in searches:
        #    for n in s.get_best():
        #        best_sents = torch.stack([n[1].sequence[1:].cpu()])
        print('n best sents', best_sents)

        # concatenates a sequence of tensors, gets the one best here, so we should use the n-best (3 best) here
        decoded_batch = best_sents.numpy()

        output_sentences = [decoded_batch[row, :] for row in range(decoded_batch.shape[0])]

        # __QUESTION 6: What is the purpose of this for loop?
        temp = list()
        for sent in output_sentences:
            first_eos = np.where(sent == tgt_dict.eos_idx)[0]  # predicts first eos token
            if len(first_eos) > 0:  # checks if the first eos token is not the beginning (position 0)
                temp.append(sent[:first_eos[0]])
            else:
                temp.append(sent)
        output_sentences = temp

        # Convert arrays of indices into strings of words
        output_sentences = [tgt_dict.string(sent) for sent in output_sentences]

        # here: adapt so that it takes the 3-best (aka n-best), % used for no overflow
        for ii, sent in enumerate(output_sentences):
            # all_hyps[int(sample['id'].data[ii])] = sent
            # variant for 3-best
            all_hyps[(int(sample['id'].data[int(ii / 3)]), int(ii % 3))] = sent

    # Write to file (write 3 best per sentence together)
    if args.output is not None:
        with open(args.output, 'w') as out_file:
            for sent_id in range(len(all_hyps.keys())):
                # variant for 1-best
                # out_file.write(all_hyps[sent_id] + '\n')
                # variant for 3-best
                out_file.write(all_hyps[(int(sent_id / 3), int(sent_id % 3))] + '\n')


def normalize(length):
    return (np.power(5 + length, args.alpha)) / (np.power(6, args.alpha))


def diverse(nodeeval, j):
    # implements diversity search from paper
    return nodeeval -args.gamma * j


if __name__ == '__main__':
    args = get_args()
    main(args)
