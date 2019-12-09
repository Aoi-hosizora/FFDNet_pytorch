# FFDNet_pytorch
A PyTorch implementation of a denoising network called FFDNet

### Dataset

+ [Waterloo Exploration Database](https://ece.uwaterloo.ca/~k29ma/exploration/)

### Usage

+ Train

```bash
python3 ffdnet.py \
    --is_train \
    --train_path './train_data/' \
    --model_path './models/' \
    --batch_size 768 \
    --epoches 80 \
    --patch_size 32 \
    --save_checkpoints 1
```

+ Test

```bash
python3 ffdnet.py \
    --is_test \
    --test_path './test_data/color.png' \
    --model_path './models/' \
    --add_noise
    --noise_sigma 30
```