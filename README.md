# Installation
1. Create conda environment and activate environment.
```bash
  conda create -n ner_env python=3.8.10 
  conda activate ner_env
```
2. Install pytorch with appropriate CUDA version.
```bash
  pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```
3. Install libraries.
```bash
  pip install -r requirement.txt
```
# Download model folder
```
Newest version: https://s.net.vn/Zmsh
```

# Train
```bash
  bash train.sh
```

# Inference
```bash
  python src/inference.py -m [Model path] -t [Text]
```

# Model performance on test set
| | precision| recall   |f1     | num samples|
| :-------- | :------- | :------- |:------|:------|
| B-MAT | 0.6386    | 0.9298    | 0.7571 | 285 |
| I-MAT | 0.9627    | 0.8431    | 0.8990 | 153 |
| B-COLOR | 0.3832 | 0.9535 | 0.5467 | 43 |
| I-COLOR | 1.0000 | 0.8000 | 0.8889 | 5  |
| Average | 0.7217 | 0.9033 | 0.7845 | 486 |
| Overall | 0.9177 | 0.9406 | 0.9290 | 486 |

Accuracy: 0.9973

# Future works
- Update more material and color vocabularies, contexts in training data
- Add more attributes for extraction.
- Intergrate Hero Gang method to huggingface.
