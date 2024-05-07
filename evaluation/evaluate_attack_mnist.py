from __future__ import print_function
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from torch.autograd import Variable
from models.small_cnn import *
import numpy as np
import tensorflow as tf 

parser = argparse.ArgumentParser(description='PyTorch MNIST Attack Evaluation')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.3,
                    help='perturbation')
parser.add_argument('--model-path',
                    default='Adversarial-attacks-via-Sequential-Quadratic-Programming/checkpoints/smallcnn_regular/model-nn-epoch10.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--data-attack-path',
                    default='Adversarial-attacks-via-Sequential-Quadratic-Programming/data_attack/mnist_X_adv.npy',
                    help='adversarial data for white-box attack evaluation')
parser.add_argument('--data-path',
                    default='C:/Users/edoar/Desktop/PythonProjects/data/mnist_X.npy',
                    help='data for white-box attack evaluation')
parser.add_argument('--target-path',
                    default='C:/Users/edoar/Desktop/PythonProjects/data/mnist_Y.npy',
                    help='target for white-box attack evaluation')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def image_check(min_delta, max_delta, min_image_adv, max_image_adv):
    valid = 1.0
    if min_delta < - args.epsilon:
        print("1")
        print(min_delta)
        valid -= 2.0
    elif max_delta > args.epsilon:
        valid -= 2.0
        print("2")
    elif min_image_adv < 0.0:
        valid -= 2.0
    elif max_image_adv > 1.0:
        print("3")
        valid -= 2.0

    if valid > 0.0:
        return True
    else:
        return False


def eval_adv_test_whitebox(model, device, X_adv_data, X_data, Y_data):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0

    with torch.no_grad():
        for idx in range(len(Y_data)):
            # load original image
            image = np.array(np.expand_dims(X_data[idx], axis=0), dtype=np.float32)
            image = np.array(np.expand_dims(image, axis=0), dtype=np.float32)
            # load adversarial image
            image_adv = np.array(np.expand_dims(X_adv_data[idx], axis=0), dtype=np.float32)
            image_adv = np.array(np.expand_dims(image_adv, axis=0), dtype=np.float32)
            # load label
            label = np.array(Y_data[idx], dtype=np.int64)

            # check bound
            image_delta = image_adv - image
            min_delta, max_delta = image_delta.min(), image_delta.max()
            min_image_adv, max_image_adv = image_adv.min(), image_adv.max()
            valid = image_check(min_delta, max_delta, min_image_adv, max_image_adv)
            if not valid:
                print('not valid adversarial image')
                break

            # transform to torch.tensor
            data_adv = torch.from_numpy(image_adv).to(device)
            target = torch.from_numpy(label).to(device)

            # evluation
            X, y = Variable(data_adv, requires_grad=True), Variable(target)
            out = model(X)
            err_robust = (out.data.max(1)[1] != y.data).float().sum()
            robust_err_total += err_robust
    if not valid:
        print('not valid adversarial image')
    else:
        print('robust_err_total: ', robust_err_total * 1.0 / len(Y_data))


def main():
    # white-box attack
    # load model
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))

    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Salva le prime 500 immagini di test e le relative etichette
    num_images_to_save = 500
    test_images_to_save = test_images[:num_images_to_save]
    test_labels_to_save = test_labels[:num_images_to_save]  
    np.save('data/mnist_X.npy', test_images_to_save)
    np.save('data/mnist_Y.npy', test_labels_to_save)
    
    # load data
    X_adv_data = np.load(args.data_attack_path)
    X_data = np.load(args.data_path)
    Y_data = np.load(args.target_path)

    eval_adv_test_whitebox(model, device, X_adv_data, X_data, Y_data)


if __name__ == '__main__':
    main()