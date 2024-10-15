# Different digits, same hyperparameters
python squat-penalty-attack.py --true-label 4 --target-label 9 --tau 1.0 --rho 1.5 --Niter 1000
python squat-penalty-attack.py --true-label 3 --target-label 0 --tau 1.0 --rho 1.5 --Niter 1000

# Same digits, differents hyperparameters
python squat-penalty-attack.py --true-label 2 --target-label 6 --tau 1.0 --rho 1.1 --Niter 1000
# python squat-penalty-attack.py --true-label 2 --target-label 6 --tau 1.0 --rho 1.5 --Niter 1000

