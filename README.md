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
# Download model
```
Newest version: https://bom.so/GJy3Bz
```

# Train
```bash
  bash train.sh
```

# Inference
```bash
  python src/inference.py -m [Model path] -f [Excel path] -t [Text]
```

# Model performance on test set
| class     | precision| recall   |f1     |
| :-------- | :------- | :------- |:------|
| MAT       | 0.968    | 0.945    | 0.956 |
| COLOR     | 0.980    | 0.889    | 0.932 |

# Future works
- Update more material and color vocabularies, contexts in training data
- Add more attributes for extraction.
