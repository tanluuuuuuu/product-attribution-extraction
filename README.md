# Installation
Python version: 3.8.10
CUDA version: 11.8
```bash
  conda create -n ner_env python=3.8.10 
  conda activate ner_env
  pip install transformers
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install jupyterlab
```

# Download model
```
https://drive.google.com/drive/folders/1DG2cgeLSggWGFzxxSFjVhcuGB5-r4gO5?usp=sharing
```
# Start notebooks
```bash
  conda activate ner_env
  jupyter lab
```

# Train
See file [train](./notebooks/train.ipynb)

# Inference
Take a look at file [inference](./notebooks/inference.ipynb)

