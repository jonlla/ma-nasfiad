# Neural Architecture Search for Convolutional Autoencoders via Cartesian Genetic Programming
## General Information

This repository contains an implementation for the neural architecture search method proposed by <cite>[Suganuma et al. (2018)][1]</cite>. Based on Cartesian Genetic Programming the method can be used to find good Convolutional Autoencoder architectures automatically. Compared to the implementation of the authors there are the following differences:
- The authors originally implemented their method for Image Restoration. However, this package can be used for any purpose. For example, I used this package in my master thesis to search for an architecture for an industrial anomaly detection problem: <cite>[Link to demonstration notebook (in German)][2] </cite>
- Internally, this package is using keras instead of pytorch
- The package can be integrated with Weights & Biases if you want to track the results automatically and includes a warm start option.

### Install and imports

```
%cd ma-nasfiad/neuro_cgp/
!pip install .
```

```python
from autoencoder_cgp.conv_autoencoder import ConvolutionalAutoencoderEvolSearch
```
### General Usage

```python
model_search = ConvolutionalAutoencoderEvolSearch(X_train, 
                                                  fitness_eval_func="reconstruction_loss",
                                                  maximize_fitness=False) # minimize reconstruction_loss
model_search.fit()
```

In this example, the evolutionary algorithm would search for a CAE architecture that minimizes the reconstruction error (MSE) between *X_train_predict* and *X_train*, where *X_train_predict* is the output of the CAE.


The best generated architecture (called candidate or individual in the evolutionary algorithm) can be selected as follows:

```python
best_individual = model_search.best_individual
# architecture can be generated for any shape e.g. larger than the input_shape used in the search phase.
input_shape = (None, 32, 32, 1)
best_model = best_individual.generate_architecture(input_layer_shape=input_shape)
```
*generate_architecture* returns a keras model that can be used for example to predict new instances:

```python
X_val_pred = model.predict(X_val)
```

## Advanced usage
### Search Space

The *SearchConfiguration* is optional for the architecture search but allows you to specify the search space and search restrictions for the generated architectures. Furthermore, parameters for the training of the neural networks can be set via the *SearchConfiguration*.

If no search space is defined the following default search space will be used:

```python
            default_conv = {"filter_size": [1, 3, 5],
                            "num_filter": [8, 16, 32, 64, 128, 256],
                            "has_pooling_layer": [False, True]}
```

Alternatively, a custom search space can be defined as follows:

```python
custom_search_space = {"filter_size": [1, 3, 5],
                "num_filter": [32, 64, 128],
                "has_pooling_layer": [False, True]}
```
The search space can be defined inside the *SearchConfiguration*. Optionally, the maximum size of the representation layer of the autoencoder can be defined as well:

```python
demo_search_config = SearchConfiguration(max_representation_size=512,
                                        block_search_space=custom_search_space)
```
Defining a maximum representation size can work as regularization to avoid overfitting.
### Keras Training Configuration

The following arguments can control the training epochs and batch_size:
```python
EPOCHS_TRAIN = 30
BATCH_SIZE= 16
```
And can be passed into the search configuration:

```python
search_config = SearchConfiguration(max_representation_size=MAX_REP_SIZE,
                                    epochs=EPOCHS_TRAIN,
                                    batch_size=BATCH_SIZE)
```

### Arguments for the evolutionary algorithm

The following arguments influence the evolutionary algorithm:
```python
GENERATIONS = 10 # Number of generations that the evolutionary algorithm is executed.
NUM_CHILDREN = 2 # Number of architecture that are generated in each generation.
NUM_COLS = 20 #  A high value results in deep neural networks, while a low value leads more often to architectures with low depth.
```

These arguments can be defined inside the constructor:

```python
model_search = ConvolutionalAutoencoderEvolSearch(X_train, 
                                                  maximize_fitness=True,
                                                  num_generations=GENERATIONS,
                                                  num_children=NUM_CHILDREN,
                                                  num_cols=NUM_COLS,
                                                  search_config=custom_search_config)
```
### Weights & Biases integration
Your W&B API key and a project name need to be specified in the constructor via the following parameters:
```
wandb_project_name
wandb_api_key
```
The architecture search will now log the results of each generated architecture as well as its hyperparameters to your specified project.

## Example Use Case

The following Google Colab Notebook demonstrates the method for an industrial anomaly detection problem and includes more documentation:
<cite>[Link to demonstration notebook (in German)][2] </cite>


[1]: https://arxiv.org/pdf/1803.00370.pdf
[2]: https://colab.research.google.com/drive/1FeoEgn9Fgav_M6dKMIRHUqJihySL050d#scrollTo=8H6GG7yM8WSY
