from autoencoder_cgp.evolutionary_components.individual import Individual
from autoencoder_cgp.evolutionary_components.evolution import evolutionary_search
from autoencoder_cgp.evolutionary_components.configurations import SearchConfiguration, EvolutionConfiguration
from autoencoder_cgp.evolutionary_components.logger import ModelInfoLoggerFactory
from autoencoder_cgp.evolutionary_components import constants
from autoencoder_cgp.evolutionary_components.operations import generate_model_from_phenotype

import os
from sklearn.metrics import mean_squared_error


class ConvolutionalAutoencoderEvolSearch:

    def __init__(self, X_train,
                 fitness_eval_func="reconstruction_loss",
                 fitness_func_args=None,
                 maximize_fitness=True,
                 num_evaluations=1,
                 search_config=None,
                 network_type="Conv",
                 num_generations=10,
                 mutation_probability=0.1,
                 num_children=4,
                 num_rows=3,
                 num_cols=20,
                 levelback=5,
                 path=None,
                 warm_start=False,
                 wandb_api_key=None,
                 wandb_project_name=None,
                 additional_logger_info: dict = None,
                 log_keras_train=True,
                 log_model_info=True):
        """
        Implementation of an architecture search for Convolutional Autoencoder (CAE)
        based on Cartesian Genetic Programming following (Suganuma et al., 2018).

        Note: Default parameters (for mutation_proba, num_children, num_rows, num_cols, levelback) are based
        on experiments by (Suganuma et al., 2018) and might not be the the best for every dataset.
        @param num_evaluations: Default = 1. If num_evaluations > 1 and the fitness of a generated architecture
            is higher then the previous best fitness, the evaluation is repeated num_evaluations times.
            The fitness that is saved for this individual is the mean of the evaluations. Use this to find models with lower variance.
        @param additional_logger_info: Default = None. This can be used to log additional information to
            a Weights & Biases project and is otherwise not required.
        @param warm_start: If True: Individual of last search is used as a starting point for the next generations.
            Loads the individual based on parameter "path".
        @param wandb_api_key: Default = None. Specify API Key for Weights & Biases if you want to track the results of the architecture search.
        @param wandb_project_name: Default = None, Specify a Weights & Biases project name.
        @param maximize_fitness: If True fitness is maximized, else fitness is minimized.
        @param network_type: Currently only "Conv" supported.
        @param levelback: Determines the the variability of the depth of generated networks. Must be in [1, num_cols]
                if levelback = 1 generated architectures will mostly be of depth num_cols.
                 If level_back = num_cols network depths is extremely variable (Parameter for Cartesian Genetic Programming).
        @param log_keras_train: If True, Keras Standard Output is printed during the network training process
        @param log_model_info: If True, generated architecture summaries are printed to the console.
        @param X_train: Training data for the Autoencoder
        @param fitness_eval_func: Function that determines how a generated architecture is evaluated,
                i.e. the Fitness function.
                Default is the Reconstruction Loss, i.e. Mean Squared Error (MSE).
                The best architecture is chosen based on the best MSE on the Train Data.
        @param fitness_func_args: Arguments for the fitness_eval_func.
        @param search_config: Search Configuration that determines additional parameters of the search
                and the architecture search space.
        @param num_generations: Number of generations.
        @param mutation_probability: Probability that a layer and the connection of layer is mutated. Must be in (0, 1].
        @param num_children: Number of children that are generated via mutation at each generation.
               Children are mutated based on previous Individual with the best fitness.
        @param num_rows:  Determines the variability of the generated networks (Parameter for Cartesian Genetic Programming).
        @param num_cols: Determines the the maximum depth of the encoder of an autoencoder.
                        E.g. if num_cols = 10 the encoder can have 10 layers as well as the decoder,
                        which results in an autoencoder with 20 layers (Parameter for Cartesian Genetic Programming).
        @param path: This path specifies the directory where the best individual of each generation is saved.
        This is required if you want to enable warm_start.
        """

        self.log_keras_train = log_keras_train
        if wandb_api_key is None:
            self.logger_factory = ModelInfoLoggerFactory(network_type, log_keras_train=log_keras_train,
                                                         log_model_info=log_model_info)
        else:
            assert wandb_project_name is not None, "If you want to use W&B also specify a project name."
            self.logger_factory = ModelInfoLoggerFactory(network_type, wandb_api_key=wandb_api_key,
                                                         wandb_project=wandb_project_name, num_cols=num_cols,
                                                         log_keras_train=log_keras_train, log_model_info=log_model_info)

        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                dir_content = os.listdir(path)
                if warm_start is False:
                    assert len(
                        dir_content) == 0, "Please choose a different path as this path contains saved individuals " \
                                           "which would be overwritten by this search. Or enable warm_start if you want to continue an existing serach. Path: " + path
        self.path = path
        self.maximize_fitness = maximize_fitness
        self.X_train = X_train
        if fitness_eval_func == "reconstruction_loss":
            self.fitness_eval_func = self._evaluate_via_recon_loss
            self.func_args = {"X_train": X_train}
        else:
            self.fitness_eval_func = fitness_eval_func
            self.func_args = fitness_func_args
        if search_config is None:
            self.search_config = SearchConfiguration(network_type=network_type)
        else:
            self.search_config: SearchConfiguration = search_config
            assert search_config.network_type == network_type, \
                "You defined a custom SearchConfiguration but specified a different network_type. " \
                f"\nIn Search Constructor: {network_type}" \
                f"\nIn Search Configuration: {search_config.network_type}"

        hyperparam_dict = {
            "batch_size": self.search_config.batch_size,
            "num_cols": num_cols,
            "num_rows": num_rows,
            "level_back": levelback,
            "mutation_proba": mutation_probability
        }

        self.logger_factory.add_hyperparams(hyperparam_dict)
        self.logger_factory.add_hyperparams(additional_logger_info)

        self.num_generations = num_generations
        self.mutation_probability = mutation_probability
        self.num_children = num_children
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.best_model: "tensorflow.keras.models.Model" = None
        self.best_individual: Individual = None
        self.levelback = levelback
        self.start_generation = 0
        self.generation_history = None
        self.fitness_history = None
        self.best_fitness_history = None
        self.start_individual = None
        self.num_evaluations = num_evaluations

        if warm_start is True:
            assert path is not None, "Warm start requires a path to find previous individuals and evolutionary config."
            evol_config = _load_evol_config(path)
            start_individual = _load_last_individual(path, evol_config)
            self.start_generation = start_individual.generation + 1
            self.start_individual = start_individual
            self.generation_history = evol_config.generation_history
            self.fitness_history = evol_config.fitness_history
            self.best_fitness_history = evol_config.best_fitness_history

        self.input_shape = (None, X_train.shape[1], X_train.shape[2], X_train.shape[3])


    def fit(self):
        best_individual, history = evolutionary_search(self.X_train,
                                                       fitness_eval_func=self.fitness_eval_func,
                                                       fitness_func_args=self.func_args,
                                                       maximize_fitness=self.maximize_fitness,
                                                       search_config=self.search_config,
                                                       num_generations=self.num_generations,
                                                       mutation_probability=self.mutation_probability,
                                                       num_children=self.num_children, num_rows=self.num_rows,
                                                       num_cols=self.num_cols, levelback=self.levelback,
                                                       path=self.path, start_generation=self.start_generation,
                                                       start_individual=self.start_individual,
                                                       fitness_history=self.fitness_history,
                                                       best_fitness_history=self.best_fitness_history,
                                                       generation_history=self.generation_history,
                                                       logger_factory=self.logger_factory,
                                                       input_layer_shape=self.input_shape,
                                                       num_evaluations=self.num_evaluations)
        self.best_model = best_individual.model
        self.best_individual = best_individual
        return history

    def _evaluate_via_recon_loss(self, model, X_train):
        X_pred = model.predict(X_train)
        return self._mean_recon_loss(X_train, X_pred)

    @property
    def best_architecture(self):
        return generate_model_from_phenotype(self.best_individual.phenotype, self.input_shape)

    @staticmethod
    def _mean_recon_loss(X_true, X_pred):
        X_true_flat = X_true.reshape(X_true.shape[0], -1)
        X_pred_flat = X_pred.reshape(X_true.shape[0], -1)

        return mean_squared_error(X_true_flat, X_pred_flat)


def _load_last_individual(path, evol_config: EvolutionConfiguration) -> "Individual":
    return Individual.load(path, evol_config.individual_filename, evol_config.filename_model)


def _load_evol_config(path) -> "EvolutionConfiguration":
    return EvolutionConfiguration.load(path + "/" + constants.EVOL_CONFIG_NAME)
