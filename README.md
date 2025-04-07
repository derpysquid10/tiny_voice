<div align="center">
  
# Tiny Workshop
Fine-tuning of Whisper Models on Edge Device CPUs
</div>
<p align="center">
   <img src="figures/tiny_workshop_banner.jpg" width="800" title="hover text">

</p>
<br />




## Setup
We set up a virtual environment using conda, and our code is developed in Python 3.12.

```bash
# Cloning the repository
git clone https://github.com/derpysquid10/tiny_workshop.git

cd tiny_workshop

# Setting up environment and installing libraries
conda env create -f environment.yml
conda activate tiny-workshop
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # We want the CPU version of pytorch
pip install -r requirements.txt
pip install peft
```

## Quickstart Example

This section provides an example of using Tiny Voice to fine-tune OpenAI's Whisper Base model on the Afrispeech-200 dataset.

We first have to load the dataset:
```bash
python tiny_workshop/data_processing.py --process-data
```

The example.py script fine-tunes the model using one of partial fine-tuning, LoRA, or IA3 on one of 3 configs of the dataset
```bash
python tiny_workshop/example.py
```

If an error ```version `GLIBCXX_3.4.30' not found``` is encountered, run ```conda install -c conda-forge libstdcxx-ng=12``` and re-run the baseline tests.




