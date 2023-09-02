import argparse

# dataset name 
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'NeuMF-end'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']
'''
ChatGPT-generated description of each of the models:
MLP: MLP stands for Multi-Layer Perceptron. It is a type of feedforward neural network that consists of multiple layers of neurons, including an input layer, one or more hidden layers, and an output layer. In the context of recommender systems, the MLP model is used to capture complex interactions between user and item features. It takes user and item embeddings as input and processes them through fully connected layers with non-linear activation functions, such as ReLU, to learn the interactions and make predictions.

GMF: As mentioned earlier, GMF stands for Generalized Matrix Factorization. It is a recommendation algorithm that extends traditional matrix factorization methods by incorporating non-linear transformations of latent factors. The GMF model uses embedding layers to represent user and item IDs as latent vectors. It then applies element-wise multiplication or other non-linear operations on these vectors to capture the interactions between users and items. GMF is designed to capture non-linear patterns and relationships in the data.

NeuMF-end: NeuMF stands for Neural Matrix Factorization. NeuMF-end is a variant of the NeuMF model that is trained end-to-end. NeuMF combines the strengths of both GMF and MLP models by integrating their predictions in a unified architecture. It takes user and item embeddings as input and processes them through separate GMF and MLP sub-networks. The outputs of these sub-networks are then combined and passed through additional layers to generate the final prediction. NeuMF-end refers to the complete end-to-end training of the NeuMF model.

NeuMF-pre: NeuMF-pre is another variant of the NeuMF model. It involves pre-training the GMF and MLP components separately and then combining them during the prediction phase. In this case, the GMF and MLP sub-networks are trained independently using pre-training techniques, such as traditional matrix factorization for GMF and standard MLP training for the MLP model. During prediction, the outputs of the pre-trained models are combined to make recommendations.
'''

# paths
main_path = './data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", 
        type=float, 
        default=0.001, 
        help="learning rate")
    parser.add_argument("--dropout", 
        type=float,
        default=0.0,  
        help="dropout rate")
    parser.add_argument("--batch_size", 
        type=int, 
        default=256, 
        help="batch size for training")
    parser.add_argument("--epochs", 
        type=int,
        default=20,  
        help="training epoches")
    parser.add_argument("--top_k", 
        type=int, 
        default=10, 
        help="compute metrics@top_k")
    parser.add_argument("--factor_num", 
        type=int,
        default=32, 
        help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", 
        type=int,
        default=3, 
        help="number of layers in MLP model")
    parser.add_argument("--num_ng", 
        type=int,
        default=4, 
        help="sample negative items for training")
    parser.add_argument("--test_num_ng", 
        type=int,
        default=99, 
        help="sample part of negative items for testing")
    parser.add_argument("--out", 
        default=True,
        help="save model or not")
    parser.add_argument("--gpu", 
        type=str,
        default="0",  
        help="gpu card ID")
    parser.add_argument("--n_folds",
        type=int,
        default=5,
        help="number of folds for cross validation")
    parser.add_argument("--fold", 
        type=int,
        default=-1,  
        help="Determine which fold should be removed for validation. 0-indexed. -1 means no fold is removed.")
    parser.add_argument("--input_path", 
        type=str,
        default="/home/jaehyung/workspace/multi_anno/datasets/ghc_annotations.npy",  
        help="path of input annotation matrix")
    parser.add_argument("--output_path", 
        type=str,
        default="./test_imputed.npy",  
        help="path of output imputed annotation matrix")

    return parser.parse_args()
