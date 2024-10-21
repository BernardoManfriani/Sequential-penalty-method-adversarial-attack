# Different digits, same tau
python spm-attack-mnist.py --true-label 4 --target-label 9 --tau 1.0 --rho 1.5
python spm-attack-mnist.py --true-label 3 --target-label 0 --tau 1.0 --rho 1.5

# Same digits, differents tau
python spm-attack-mnist.py --true-label 8 --target-label 1 --tau 1.0 --rho 1.1
python spm-attack-mnist.py --true-label 8 --target-label 1 --tau 1.0 --rho 1.5

# Different datsets
python spm-attack-imagenet.py
