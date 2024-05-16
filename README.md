<h1 align="center">Welcome to WavBERT: Exploiting Semantic and Non-semantic Speech using Wav2vec and BERT for Dementia Detection 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="http://www.homepages.ed.ac.uk/sluzfil/ADReSSo-2021/" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="https://github.com/kefranabg/readme-md-generator/graphs/commit-activity" target="_blank">
    <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" />
  </a>
</p>

> In this project, we exploit semantic and non-semantic information from patient’s speech data usingWav2vec and Bidirectional Encoder Representations from Transformers (BERT) for dementia detection. We first propose a basic WavBERT model by extracting semantic information from speech data using Wav2vec, and analyzing the semantic information using BERT for dementia detection. While the basic model discards the non-semantic information, we propose extended WavBERT models that convert the output ofWav2vec to the input to BERT for preserving the non-semantic information in dementia detection. Specifically, we determine the locations and lengths of inter-word pauses using the number of blank tokens from Wav2vec where the threshold for setting the pauses is automatically generated via BERT. We further design a pre-trained embedding conversion network that converts the output embedding of Wav2vec to the input embedding of BERT, enabling the fine-tuning of WavBERT with non-semantic information. Our evaluation results using the ADReSSo dataset showed that the WavBERT models achieved the highest accuracy of 83.1% in the classification task, the lowest Root-Mean-Square Error (RMSE) score of 4.44 in the regression task, and a mean F1 of 70.91% in the progression task. We confirmed the effectiveness of WavBERT models exploiting both semantic and non-semantic speech.

### 🏠 [Homepage](https://github.com/billzyx/WavBERT)

## Dataset Results

<img src="https://github.com/billzyx/WavBERT/blob/master/Wav2Vec.png"
     alt="WavBert Data"
     style="float: center; margin-right: 10px;" />

## Author

👤 **Xiaohui Liang**

- Website: http://faculty.umb.edu/xiaohui.liang/
- Website: https://www.linkedin.com/in/xiaohui-liang-7622a419/

## Author

👤 **Youxiang Zhu**

- Website: https://billzyx.github.io/
- GitHub: [@ billzyx ](https://github.com/billzyx)

## Author

👤 **Abdelrahman Obyat**

- Website: https://www.linkedin.com/in/abdelrahman-obyat-52065b173/
- GitHub: [@ obyat ](https://github.com/obyat)

### The challenge homepage can be found here:

```
http://www.homepages.ed.ac.uk/sluzfil/ADReSSo-2021/
```

# Installation

Update 22-10-20: fit recent dependency change

Update 24-05-16: Need to downgrade numpy to 1.19 to fit the old dependency

## Basic dependencies (root required)

```shell
sudo apt-get install apt-utils gcc libpq-dev libsndfile-dev
sudo apt-get install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
sudo apt-get install libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev
```

## Python dependencies (using conda)

```shell
conda create --name wavbert python=3.8 -y
conda activate wavbert
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

pip install fairseq==0.10.2 transformers==4.18.0 datasets==2.1.0
pip install matplotlib tqdm librosa editdistance sentencepiece jiwer
# Downgrade numpy to 1.19
pip install numpy==1.19 numba==0.51.0 librosa==0.8.0 resampy==0.2.2 pandas==1.0.5
```

## Clone WavBERT project

```shell
git clone https://github.com/billzyx/WavBERT.git
cd WavBERT

export WavBERT_PATH=/path/to/WavBERT

mkdir external_lib
cd external_lib
```

## Install Kenlm

```shell
git clone https://github.com/kpu/kenlm.git
cd kenlm
git checkout d70e28403f07e88b276c6bd9f162d2a428530f2e
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=20 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j 16
export KENLM_ROOT_DIR=$WavBERT_PATH'/external_lib/kenlm/'
cd ../..
```

## Install Wav2letter

```shell
# Optional
export USE_CUDA=0

git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git
cd wav2letter/bindings/python
pip install -e .
cd ../../../..
```

## Download pre-training weights

```shell
cd pre_train_weights
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt
cd ..
```

## Test Wav2vec ASR inference

```shell
CUDA_VISIBLE_DEVICES=0 python recognize.py --wav_path /path/to/xxx.wav
```


## Acess to the DataSet can be requested below:

```
For Diagnosis task (train and test):   
https://media.talkbank.org/dementia/English/0extra/ADReSSo21-diagnosis-train.tgz
https://media.talkbank.org/dementia/English/0extra/ADReSSo21-diagnosis-test.tgz 
For Progression Tasks (train and test):     
https://media.talkbank.org/dementia/English/0extra/ADReSSo21-progression-train.tgz
https://media.talkbank.org/dementia/English/0extra/ADReSSo21-progression-test.tgz
```

## Procedures for training:

```
# Get Wav2vec embedding
python3 audio_asr_to_text.py

# Get pause thresholds
python3 check_pause_length.py

# Train WavBERT model (parameters may be needed, check text_train.py)
python3 text_train.py
# Or
sh run2.sh
```


## Procedures for pre-training:

```
# Download LibriSpeech dataset, merge train-clean-100 train-clean-360 train-other-500 to train_960

# Get Wav2vec embedding of LibriSpeech dataset
python3 pre_train_audio_asr_to_text.py

# Pre-train WavBERT model (ASR embedding conversion part)
sh run.sh
```

## Show your support

Give a ⭐️ if this project helped you!

---

_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_
