# Adversarial Attacks via Sequential Quadratic Programming

Implementation of the **SQUAT** Algorithm. (Théo Beuzeville, Alfredo Buttari, Serge Gratton, Theo Mary, Erkan Ulker. Adversarial Attacks via Sequential Quadratic Programming. 2022.)

## Prerequisites
```
$ pip install -r requirements.txt
```

## Train Robust SmallCNN
### Regular Training on MNIST
```bash
$ python train_regular_mnist.py \
  --batch-size 128 \
  --test-batch-size 128 \
  --epochs 20 \
  --lr 0.01 \
  --momentum 0.9 \
  --epsilon 0.3 \
  --num-steps 40 \
  --step-size 0.01 \
  --beta 1.0 \
  --seed 1 \
  --log-interval 100 \
  --model-dir 'Adversarial-attacks-via-Sequential-Quadratic-Programming\checkpoints\smallcnn_regular' \
  --save-freq 10
```

### Adversarial Training by TRADES
```bash
$ python train_trades_mnist.py \
  --batch-size 128 \
  --test-batch-size 128 \
  --epochs 20 \
  --lr 0.01 \
  --momentum 0.9 \
  --epsilon 0.3 \
  --num-steps 40 \
  --step-size 0.01 \
  --beta 1.0 \
  --seed 1 \
  --log-interval 100 \
  --model-dir 'Adversarial-attacks-via-Sequential-Quadratic-Programming\checkpoints\smallcnn_trades' \
  --save-freq 10
```

### Adversarial Training by DDN
```bash
$ python train_ddn_mnist.py \
  --batch-size 128 \
  --test-batch-size 128 \
  --epochs 20 \
  --lr 0.01 \
  --momentum 0.9 \
  --epsilon 0.3 \
  --num-steps 40 \
  --step-size 0.01 \
  --beta 1.0 \
  --seed 1 \
  --log-interval 100 \
  --model-dir 'Adversarial-attacks-via-Sequential-Quadratic-Programming\checkpoints\smallcnn_ddn' \
  --save-freq 10
```

## Attack the SmallCNN
### SQUAT attack
```bash
$ python main.py 
```

## Robustness Evaluation
    1. Download `mnist_X.npy` and `mnist_Y.npy`.
    2. Run your own attack on `mnist_X.npy` and save your adversarial images as `mnist_X_adv.npy`.
    3. Put `mnist_X_adv.npy` under `./data_attack`.
    4. Run the evaluation code:
```bash
$ python evaluate_attack_mnist.py
```

## Reference
For technical details and full experimental results, please check the paper ([https://hal.archives-ouvertes.fr/hal-03752184](https://hal.archives-ouvertes.fr/hal-03752184))