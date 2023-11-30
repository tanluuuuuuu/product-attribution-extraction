# Installation
Python version: 3.8.10

CUDA version: 11.8
```bash
  conda create -n ner_env python=3.8.10 
  conda activate ner_env
  pip install -r requirement.txt
```

# Download model
```
V5: https://drive.google.com/drive/folders/1IIVdpPzj2FjWY0800-ZaqP7vpPA8akLG?usp=drive_link
```
# Start notebooks
```bash
  conda activate ner_env
  jupyter lab
```

# Train
```bash
  python src/train.py
```
See file [train](./notebooks/train.ipynb) for more information

# Inference
```bash
  python src/inference.py -m [Link model] -t [Text]
```
Or take a look at file [inference](./notebooks/inference.ipynb) for more information

# Performance 
| class     | precision| recall   |f1     |
| :-------- | :------- | :------- |:------|
| MAT       | 0.968    | 0.945    | 0.956 |
| COLOR     | 0.980    | 0.889    | 0.932 |

# Note
For better result, please remove brand name from text due to  potential contextual similarities.

# Future works
- Update more material vocabularies, contexts in training data

