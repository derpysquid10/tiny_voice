<p align="center">
   <img src="figures/tiny_voice_banner.jpg" width="800" title="hover text">

</p>
<br />

<div align="center">

# Tiny Voice 
By Tiny Workshop<br/>
An open-source, user friendly library to fine-tune speech-to-text models on edge device CPUs. <br/>


![Static Badge](https://img.shields.io/badge/python-3.12-blue?style=flat&logo=Python)
[![Contributors](https://img.shields.io/github/contributors/derpysquid10/tiny_voice?style=flat&logo=github)](https://github.com/derpysquid10/tiny_voice/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/derpysquid10/tiny_voice?style=flat&logo=target)](https://github.com/derpysquid10/tiny_voice/issues)
[![Last Commit](https://img.shields.io/github/last-commit/derpysquid10/tiny_voice?style=flat&logo=git)](https://github.com/derpysquid10/tiny_voice/commits)
[![Tiny Voice Paper](https://img.shields.io/badge/Tiny%20Voice-Paper-red?style=flat&logo=carrd)](https://github.com/derpysquid10/tiny_voice/blob/main/Tiny_Voice.pdf)





</div>


## Installation
We set up a virtual environment using conda, and our code is developed in Python 3.12.<br/>
**NOTE:** Tiny Voice is has only been tested on Linux systems, if using Windows, please install [Windows System Linux ](https://learn.microsoft.com/en-us/windows/wsl/setup/environment#set-up-your-linux-username-and-password) (WSL):

```bash
# Cloning the repository
git clone https://github.com/derpysquid10/tiny_voice.git

cd tiny_voice

# Setting up environment and installing libraries
conda env create -f environment.yml
conda activate tiny-voice
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # We want the CPU version of pytorch
pip install -r requirements.txt
```

## Quickstart Example

This section provides an example of using Tiny Voice to fine-tune OpenAI's Whisper Base model on the Afrispeech-200 dataset.

We first have to load the dataset:
```bash
python tiny_voice/data_processing.py --process-data
```

The example.py script fine-tunes the model using one of partial fine-tuning, LoRA, or IA3 on one of 2 configs of the dataset
```bash
python tiny_voice/example.py
```

If an error ```version `GLIBCXX_3.4.30' not found``` is encountered, run ```conda install -c conda-forge libstdcxx-ng=12``` and re-run the baseline tests.





