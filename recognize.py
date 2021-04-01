import os

use_gpu_num = '0'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num

import torch
import argparse
import librosa
import torch.nn.functional as F
import itertools as it
from fairseq import utils
from fairseq.models import BaseFairseqModel
from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
from fairseq.data import data_utils
from fairseq.models.wav2vec.wav2vec2_asr import base_architecture, Wav2VecEncoder
from wav2letter.decoder import CriterionType
from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes
import re
import numpy as np

from collections import Counter
from multiprocessing import Pool

import torch
from fairseq import utils
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line
import contextlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(t, bpe_symbol, escape_unk, extra_symbols_to_ignore)
                for t in tensor
            )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        # extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]

        # if hasattr(self, "bos_index"):
        #     extra_symbols_to_ignore.add(self.bos())

        sent = " ".join(
            token_string(i)
            for i in tensor
            if utils.item(i) not in extra_symbols_to_ignore
        )

        return data_utils.post_process(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))
            )
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

        self.pad_to_multiple_(padding_factor)

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(PathManager.get_local_path(f), "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file.".format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            PathManager.mkdirs(os.path.dirname(f))
            with PathManager.open(f, "w", encoding="utf-8") as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print("{} {}".format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(
            f,
            zip(
                ex_keys + self.symbols[self.nspecial :],
                ex_vals + self.count[self.nspecial :],
            ),
        )

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(
        filename, tokenize, eos_word, worker_id=0, num_workers=1
    ):
        counter = Counter()
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Dictionary._add_file_to_dictionary_single_worker,
                        (filename, tokenize, dict.eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_file_to_dictionary_single_worker(
                    filename, tokenize, dict.eos_word
                )
            )


class Wav2VecEncoder(Wav2VecEncoder):
    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        encoder_out_no_proj = x
        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
            'encoder_out_no_proj': encoder_out_no_proj,
        }


class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, w2v_encoder, args):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, target_dict):
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = Wav2VecEncoder(args, target_dict)
        return cls(w2v_encoder, args)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


class W2lDecoder(object):
    def __init__(self, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = 1

        self.criterion_type = CriterionType.CTC
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        self.asg_transitions = None

    def generate(self, model, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(model, encoder_input)
        # print(emissions)
        # print(emissions.size())
        return self.decode(emissions)

    def get_emissions(self, model, encoder_input):
        """Run encoder and normalize emissions"""
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = model(**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            emissions = model.get_normalized_probs(encoder_out, log_probs=True)

        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        # print(idxs)
        # idxs = (g[0] for g in it.groupby(idxs))
        # new_idxs = []
        # for g in it.groupby(idxs):
        #     if g[0] != 4 and g[0] != 0:
        #         new_idxs.append(g[0])
        #     else:
        #         new_idxs.extend(list(g[1]))
        # idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = list()

        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)

        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}] for b in range(B)
        ]


def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == 'wordpiece':
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == 'letter':
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol == 'letter_b':
        sentence = sentence.strip().replace(" ", "")
        sentence = sentence.replace("<s><s><s><s><s><s><s><s><s><s><s><s><s><s>",
                                    "<s><s><s><s><s><s><s><s><s><s><s><s><s><s>|")
        sentence = re.sub('\\b(<s>)+\\b', '', sentence)
        sentence = re.sub('\\b(<s>)+\'', '\'', sentence)
        sentence = sentence.replace("<s>", "|")
        while sentence.startswith('|'):
            sentence = sentence[1:]
        # sentence = re.sub('\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|'
        #                   '\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|+', '....... ', sentence)
        # sentence = re.sub('\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|+', '..... ', sentence)
        # sentence = re.sub('\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|+', '... ', sentence)
        # sentence = re.sub('\|\|\|\|\|\|\|\|\|\|+', '. ', sentence)
        # sentence = re.sub('\|\|\|\|\|+', ', ', sentence)
        sentence = re.sub('\|+', ' ', sentence)
    elif symbol is not None and symbol != 'none':
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence


def get_feature(filepath):
    def postprocess(feats, sample_rate):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats

    wav, sample_rate = librosa.load(filepath, sr=16000)
    feats = torch.from_numpy(wav).float()
    feats = postprocess(feats, sample_rate)
    return feats


def load_model(model_path, target_dict):
    w2v = torch.load(model_path)
    model = Wav2VecCtc.build_model(w2v["args"], target_dict)
    model.load_state_dict(w2v["model"], strict=True)
    model.to(device)

    return model


class ASR:
    def __init__(self, model_weight, target_dict):
        self.target_dict = Dictionary.load(target_dict)
        self.model = load_model(model_weight, self.target_dict)
        self.model.eval()
        self.generator = W2lViterbiDecoder(self.target_dict)

    def predict_file(self, file_path):
        generator = W2lViterbiDecoder(self.target_dict)
        sample = dict()
        net_input = dict()
        feature = get_feature(file_path)
        net_input["source"] = feature.unsqueeze(0).to(device)

        padding_mask = torch.BoolTensor(net_input["source"].size(1)).fill_(False).unsqueeze(0).to(device)

        net_input["padding_mask"] = padding_mask
        sample["net_input"] = net_input

        with torch.no_grad():
            hypo = generator.generate(self.model, sample, prefix_tokens=None)
            # print(hypo[0][0]["tokens"].size())

        hyp_pieces = self.target_dict.string(hypo[0][0]["tokens"].int().cpu(), bpe_symbol='none')
        asr_result = post_process(hyp_pieces, 'none')
        # print(asr_result)

        audio_embedding = self.model(**net_input)
        audio_embedding = audio_embedding['encoder_out_no_proj'].squeeze(1).cpu().numpy()
        # audio_embedding = audio_embedding['encoder_out'].squeeze(1).detach().cpu().numpy()
        # print(np.shape(audio_embedding))
        return asr_result, audio_embedding


def main():
    parser = argparse.ArgumentParser(description='Wav2vec-2.0 Recognize')
    parser.add_argument('--wav_path', type=str,
                        default='~/xxx.wav',
                        help='path of wave file')
    parser.add_argument('--w2v_path', type=str,
                        default='pre_train_weights/wav2vec_vox_960h_pl.pt',
                        help='path of pre-trained wav2vec-2.0 model')
    parser.add_argument('--target_dict_path', type=str,
                        default='pre_train_weights/dict.ltr.txt',
                        help='path of target dict (dict.ltr.txt)')
    args = parser.parse_args()
    sample = dict()
    net_input = dict()

    feature = get_feature(args.wav_path)
    target_dict = Dictionary.load(args.target_dict_path)

    model = load_model(args.w2v_path, target_dict)
    model.eval()

    generator = W2lViterbiDecoder(target_dict)
    net_input["source"] = feature.unsqueeze(0).to(device)

    padding_mask = torch.BoolTensor(net_input["source"].size(1)).fill_(False).unsqueeze(0).to(device)

    net_input["padding_mask"] = padding_mask
    sample["net_input"] = net_input

    with torch.no_grad():
        hypo = generator.generate(model, sample, prefix_tokens=None)
        print(hypo[0][0]["tokens"].size())

    hyp_pieces = target_dict.string(hypo[0][0]["tokens"].int().cpu(), bpe_symbol='none')
    print(len(hyp_pieces.split()))
    print(post_process(hyp_pieces, 'letter_b'))


if __name__ == '__main__':
    main()