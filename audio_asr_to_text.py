import os
from tqdm import tqdm
from recognize import ASR
import numpy as np


def process_path(path, audio_dir_name='', remove_name=''):
    asr = ASR(model_weight='pre_train_weights/wav2vec_vox_960h_pl.pt', target_dict='pre_train_weights/dict.ltr.txt')
    for name in tqdm(sorted(os.listdir(path))):
        if name.endswith('.wav'):
            file_path = os.path.join(path, name)
            text, audio_embedding = asr.predict_file(file_path)
            new_file_path = file_path.replace(audio_dir_name, 'asr_text') \
                .replace(remove_name, '').replace('.wav', '.txt')
            if not os.path.isdir(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))
            with open(new_file_path, 'w') as f:
                f.write(text)

            new_file_path = file_path.replace(audio_dir_name, 'asr_embedding') \
                .replace(remove_name, '').replace('.wav', '.npy')
            if not os.path.isdir(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))
            np.save(new_file_path, audio_embedding)


def process_20():
    full_wave_enhanced_audio_path = '../ADReSS/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc'
    process_path(full_wave_enhanced_audio_path, audio_dir_name='Full_wave_enhanced_audio', remove_name='../ADReSS/')
    full_wave_enhanced_audio_path = '../ADReSS/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd'
    process_path(full_wave_enhanced_audio_path, audio_dir_name='Full_wave_enhanced_audio', remove_name='../ADReSS/')
    full_wave_enhanced_audio_path = '../ADReSS/ADReSS-IS2020-data/test/Full_wave_enhanced_audio'
    process_path(full_wave_enhanced_audio_path, audio_dir_name='Full_wave_enhanced_audio', remove_name='../ADReSS/')


def process_21():
    full_wave_enhanced_audio_path = 'ADReSSo21/diagnosis/train/audio/ad'
    process_path(full_wave_enhanced_audio_path, audio_dir_name='audio', remove_name='')
    full_wave_enhanced_audio_path = 'ADReSSo21/diagnosis/train/audio/cn'
    process_path(full_wave_enhanced_audio_path, audio_dir_name='audio', remove_name='')

    full_wave_enhanced_audio_path = 'ADReSSo21/progression/train/audio/decline'
    process_path(full_wave_enhanced_audio_path, audio_dir_name='audio', remove_name='')
    full_wave_enhanced_audio_path = 'ADReSSo21/progression/train/audio/no_decline'
    process_path(full_wave_enhanced_audio_path, audio_dir_name='audio', remove_name='')

    full_wave_enhanced_audio_path = 'ADReSSo21/diagnosis/test-dist/audio'
    process_path(full_wave_enhanced_audio_path, audio_dir_name='audio', remove_name='')

    full_wave_enhanced_audio_path = 'ADReSSo21/progression/test-dist/audio'
    process_path(full_wave_enhanced_audio_path, audio_dir_name='audio', remove_name='')


if __name__ == '__main__':
    process_21()
