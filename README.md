<div align="center">
  
# Tiny Workshop
Fine-tuning of Whisper Models on Edge Device CPUs
</div>
<p align="center">
   <img src="figures/tiny_workshop_banner.jpg" width="800" title="hover text">

</p>
<br />


## TODO
- [ ] requirements.txt![Uploading tiny_workshop_banner.jpg…]()

- [ ] python environment (python 3.10?), set up mamba over conda
- [x] cookie cutter file format
- [ ] setup instructions, download data instructions, etc
- [ ] notebooks for testing, %autoreload
- [ ] models folder: predictions, training log, model config, eval stats
- [ ] fnpg command line for manipulating audio and video, installable wiht python with mamba
- [ ] make svg figures first, then convert to png. inkscape
- [ ] 

## Setup
```bash
# Cloning the repository
git clone ** **ADD REPOSITORY LINK **

cd tiny_workshop

# Setting up environment and installing libraries
conda env create -f environment.yml
conda activate tiny-workshop
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu   # We want CPU version of pytorch
pip install -r requirements.txt
```
