## Python3 file:


#### main.py
* Run the excuters and some configurations. (Resampling, pre-training...)
** run_nn()/ run_nn_cv(): 10-fold crossvalidation
#### Executer.py
* Executed a configuration of NN(training process...), running single NN-model.
#### torch_model.py
* Currently, CNN, MLP, RNN(LSTMcell), MultiheadAttention included.
#### Handler.py
* Encapsulated the encoded data for specific format(with stratified distribution). e.g pytorch: tensor/Sklearn: numpy array
#### Encoder.py
* Encoded the text corpus.
#### Dataset.py
* Achieve the paper dataset.
