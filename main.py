from src import data_preparation, model_definition, evaluation, utils, squat
import torch 

def main():
    dataset = data_preparation.load_dataset()
    a = utils.get_random_image(1, dataset, seed=111) 
    b = utils.get_random_image(5, dataset, seed=111)
    x = a.clone()
    x_k = b.clone()
    # x_k = torch.rand((1,28,28)) 
    squat.squat_algorithm(x, x_k)
    
    # model = model_definition.define_model()
    # training.train_model(model)
    # evaluation.evaluate_model(model)

if __name__ == "__main__":
    main()