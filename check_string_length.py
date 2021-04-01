import os
from tqdm import tqdm
from core import load_file


def process_path(path):
    for name in (sorted(os.listdir(path))):
        if name.endswith('.txt'):
            file_path = os.path.join(path, name)
            text = load_file(file_path, level_list=(), punctuation_list=())
            print(file_path, len(text['text'].split()))
            print(text['text'].lower())


if __name__ == '__main__':
    # full_wave_enhanced_audio_path = 'ADReSS-IS2020-data/train/asr_text/cc'
    # full_wave_enhanced_audio_path = 'ADReSS-IS2020-data/train/asr_text/cd'
    # full_wave_enhanced_audio_path = 'ADReSS-IS2020-data/test/asr_text'
    # full_wave_enhanced_audio_path = 'ADReSSo21/diagnosis/train/asr_text/ad'
    # process_path(full_wave_enhanced_audio_path)
    # full_wave_enhanced_audio_path = 'ADReSSo21/diagnosis/train/asr_text/cn'
    # process_path(full_wave_enhanced_audio_path)

    full_wave_enhanced_audio_path = 'ADReSSo21/progression/train/asr_text/decline'
    process_path(full_wave_enhanced_audio_path)
    full_wave_enhanced_audio_path = 'ADReSSo21/progression/train/asr_text/no_decline'
    process_path(full_wave_enhanced_audio_path)

    # full_wave_enhanced_audio_path = 'ADReSSo21/diagnosis/test-dist/asr_text/'
    # process_path(full_wave_enhanced_audio_path)

    full_wave_enhanced_audio_path = 'ADReSSo21/progression/test-dist/asr_text/'
    process_path(full_wave_enhanced_audio_path)

