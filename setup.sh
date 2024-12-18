#!/usr/bin/env bash

## Create conda environment
conda create -n 10701-project python=3.10
conda activate 10701-project

## Modify this command depending on your system's environment.
## As written, this command assumes you have CUDA on your machine, but
## refer to https://pytorch.org/get-started/previous-versions/ for the correct
## command for your system.
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia # for Linux & Windows
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch # for OSX

pip install scikit-learn
pip install transformers
pip install pandas
pip install tqdm
pip install seaborn
pip install matplotlib
pip install langchain langchain_community 
pip install ollama # ollama client
pip install langchain_ollama # local inference and embeddings via Ollama
pip install swifter  # parallel processing for 
pip install vaderSentiment # for sentence 
