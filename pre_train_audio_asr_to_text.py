import os
from tqdm import tqdm
from recognize import ASR
import numpy as np
from pathlib import Path
import json


def process_path(data_path, output_path, extension='.flac'):
    file_list = [str(path) for path in Path(data_path).rglob('*' + extension)]
    asr = ASR(model_weight='pre_train_weights/wav2vec_vox_960h_pl.pt', target_dict='pre_train_weights/dict.ltr.txt')
    data_list = []
    for file_path in tqdm(sorted(file_list)):
        text, audio_embedding = asr.predict_file(file_path)
        new_file_path = os.path.join(output_path, 'text', os.path.basename(file_path).replace(extension, '.txt'))
        if not os.path.isdir(os.path.dirname(new_file_path)):
            os.makedirs(os.path.dirname(new_file_path))
        with open(new_file_path, 'w') as f:
            f.write(text)

        new_file_path = os.path.join(output_path, 'audio_embedding',
                                     os.path.basename(file_path).replace(extension, '.npy'))
        if not os.path.isdir(os.path.dirname(new_file_path)):
            os.makedirs(os.path.dirname(new_file_path))
        np.save(new_file_path, audio_embedding)

        data_dict = dict()
        data_dict['text'] = text
        data_dict['audio_embedding'] = os.path.abspath(new_file_path)
        data_list.append(data_dict)
    with open(os.path.join(output_path, 'data.json'), 'w') as f:
        json.dump(data_list, f)


if __name__ == '__main__':
    pre_train_audio_path = '../speech_recognition/raw_data/LibriSpeech/train_960/'
    asr_output_path = 'pre_train_data/LibriSpeech/'
    process_path(pre_train_audio_path, asr_output_path)

