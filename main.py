from src import data_preparation, model_definition, evaluation, utility_functions, squat
import torch
from models.small_cnn import SmallCNN


def main():
    dataset = data_preparation.load_dataset()
    a = utility_functions.get_random_image(1, dataset)
    b = utility_functions.get_random_image(1, dataset)
    #utility_functions.show_image(a)

    x = a.clone()
    xk = b.clone()
    # start_time = time.time()

    squat.squat_algorithm(x, xk)
    

    # model = model_definition.define_model()
    # training.train_model(model)
    # evaluation.evaluate_model(model)

if __name__ == "__main__":
    main()