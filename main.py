from src import data_preparation, model_definition, evaluation, utils, squat, sequentialPenaltyMethod, SPM2
import config

def main():
    dataset = data_preparation.load_dataset()
    x = utils.get_random_image(config.target_class, dataset, seed=111)
    N_iter = config.N_iter
    N_1 = config.N_1
    α = config.α
    β = config.β
    j = config.target_class
    x_j = utils.get_random_image(config.target_class, dataset, seed=111)
    x = utils.get_random_image(config.original_class, dataset, seed=111) # image to attack
    
<<<<<<< Updated upstream
    squat.SQUAT(x, x_j, j, N_iter, N_1, α, β)
    
=======
    # squat.SQUAT(x, N_iter, N_1, α, β)
    print("inizio")
    # sequentialPenaltyMethod.generate_adversarial_example(x, target_class=1)
    SPM2.spm(x, 1)

>>>>>>> Stashed changes
    # model = model_definition.define_model()
    # training.train_model(model)
    # evaluation.evaluate_model(model)

if __name__ == "__main__":
    main()
    