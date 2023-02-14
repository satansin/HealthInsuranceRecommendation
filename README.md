# Automatic Personalized Health Insurance Recommendation Based on Utility and User Feedback
Code and sampled dataset for KDD 2023 submission Automatic Personalized Health Insurance Recommendation Based on Utility and User Feedback.

## Installation
The following packages or environments are required for this repository.
- python 3.7
- pytorch 1.6.0+cu101
- torchvision 0.7.0+cu101
- torch-cluster 1.5.8
- torch-scatter 2.0.5
- torch-sparse 0.6.8
- torch-spline-conv 1.2.0
- torch-geometric 1.6.3
- tensorboardx 2.1
- pandas 1.3.5
- numpy 1.21.5
- scikit-learn 1.0.2

## Running the Code
Our code in this repository includes predicting the click probabilities of the sampled user feedback dataset (which has been pre-processed) with a GraphCM model to be trained and generating the updated weights.

### Training the GraphCM Model
- Go into folder `GraphCM/`.
- Run the following script to train the model.
```
python -u run.py --train --optim adam --eval_freq 100 --check_point 1000 --dataset ../sampled_dataset --combine exp_mul --gnn_neigh_sample 0 --gnn_concat False --inter_neigh_sample 0 --learning_rate 0.001 --lr_decay 0.5 --weight_decay 1e-5 --dropout_rate 0.5 --num_steps 20000 --embed_size 64 --hidden_size 64 --batch_size 256 --patience 6 --max_d_num 50
```
- Test the trained model (optional)
```
python -u run.py --test --optim adam --eval_freq 100 --check_point 1000 --dataset ../sampled_dataset --combine exp_mul --gnn_neigh_sample 0 --gnn_concat False --inter_neigh_sample 0 --learning_rate 0.001 --lr_decay 0.5 --weight_decay 1e-5 --dropout_rate 0.5 --num_steps 20000 --embed_size 64 --hidden_size 64 --batch_size 256 --patience 6 --max_d_num 50 --load_model=20000
```
After training, the trained model will be saved in `GraphCM/outputs/`.

### Predicting the Click Probabilities
Run the following script to predict the click probabilities of the simulated user feedback dataset (which has been pre-processed).
```
python -u run.py --predict --optim adam --eval_freq 100 --check_point 1000 --dataset ../sampled_dataset --combine exp_mul --gnn_neigh_sample 0 --gnn_concat False --inter_neigh_sample 0 --learning_rate 0.001 --lr_decay 0.5 --weight_decay 1e-5 --dropout_rate 0.5 --num_steps 20000 --embed_size 64 --hidden_size 64 --batch_size 256 --patience 6 --max_d_num 50 --load_model=20000
```
The prediction results will be saved in `sampled_dataset/predict_results.txt`.

### Generating the Updated Weights
Run the following script (in the root path of this repository) to generate the updated weights based on the above prediction results.
```
python weight_update.py sampled_dataset
```
The updating rate is default to 0.1. If you would like to specify a different updating rate (e.g., 0.2), use the following script instead where the value of the specified updating rate is placed as the last argument.
```
python weight_update.py sampled_dataset 0.2
```
The updated weight results will be saved in `sampled_dataset/updated_weights-XX.csv`, where XX is the updating rate.
