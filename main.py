from src import data_preparation, model_definition, evaluation, utils, squat
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
    image_to_attack = utils.get_random_image(config.original_class, dataset, seed=111)
    
    squat.SQUAT(image_to_attack, x_j, j, N_iter, N_1, α, β)
    
    # model = model_definition.define_model()
    # training.train_model(model)
    # evaluation.evaluate_model(model)

if __name__ == "__main__":
    main()
    