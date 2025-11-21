# DReX: Pure Vision Fusion of Self-Supervised and Convolutional Representations for Image Complexity Prediction

Implementation for the preprint: _DReX: Pure Vision Fusion of Self-Supervised and Convolutional Representations for Image Complexity Prediction_.

![DRex](https://raw.githubusercontent.com/jskaza/DReX/refs/heads/main/fig1.png)

## Dependencies
```bash
matplotlib==3.10.7
numpy==2.3.5
pandas==2.3.3
Pillow==12.0.0
scipy==1.16.3
torch==2.9.0
torchvision==0.24.0
tqdm==4.67.1
transformers==4.57.1
```


## Model Checkpoint
`DReX.pth` is the model checkpoint from the paper.

## Training
```bash
python train.py
```

## Evaluation
```bash
python eval.py
```

## Dataset Used for Training
[IC9600](https://github.com/tinglyfeng/IC9600)

Tinglei Feng, Yingjie Zhai, Jufeng Yang, Jie Liang, Deng-
Ping Fan, Jing Zhang, Ling Shao, and Dacheng Tao. IC9600:
A benchmark dataset for automatic image complexity assessment. _IEEE Transactions on Pattern Analysis and Machine
Intelligence_, 45(7):8577–8593, 2022. 
