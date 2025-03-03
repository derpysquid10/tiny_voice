<div align="center">
  
# Tiny Workshop
Fine-tuning of Whisper Models on Edge Device CPUs
</div>
<p align="center">
   <img src="figures/tiny_workshop_banner.jpg" width="800" title="hover text">

</p>
<br />


## TODO
- [x] requirements.txt!
- [x] python environment 
- [x] cookie cutter file format
- [x] setup instructions, 
- [ ] data information, download data instructions, etc
- [ ] notebooks for testing, %autoreload
- [ ] models folder: predictions, training log, model config, eval stats
- [ ] fnpg command line for manipulating audio and video, installable wiht python with mamba
- [ ] make svg figures first, then convert to png. inkscape
- [ ] 

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
```


## Preparing Dataset

## Running Tiny Workshop