import tensorflow as tf

models = tf.keras.models
layers = tf.keras.layers

import wandb
import functools


class ModelInfoLogger:

    def __init__(self, wandb_enabled=False, wandb_project=None, log_keras_train=True, generation=None, child=None,
                 hyperparam_dict: dict = None, num_cols=None, log_model_info=True):
        """If no parameters are specified this class is just used to log the model info to the console.
        @param log_model_info:
        @param num_cols:
        @param wandb_project:
        """
        self.log_model_info = log_model_info
        self.num_cols = num_cols
        self.wandb_enabled = wandb_enabled

        if self.wandb_enabled:
            wandb.init(project=wandb_project, name=f"generation_{generation}_{child}", reinit=True)
            wandb.config.generation = generation
            if hyperparam_dict is not None:
                wandb.config.update(hyperparam_dict)
                if "fitness_measure_name" in hyperparam_dict.keys():
                    self.fitness_measure_name = hyperparam_dict["fitness_measure_name"]

        self.log_keras_train = log_keras_train
        if not self.log_keras_train:
            self.keras_verbose = 0
        else:
            self.keras_verbose = 2
        self.model_info = ""
        self.current_layer_index = 0

        self.layer_output_shapes = {}

    def log_custom_info(self, custom_info):
        self.model_info += custom_info

    def print_model_info_with_error(self, error):
        print("--------------------------------------------------------------------------")
        print("ORIGINAL ERROR MESSAGE")
        print(error)
        print("--------------------------------------------------------------------------")
        print("An error occurred during model creation:")
        print(self.model_info + "ERROR OCCURRED HERE")
        print("--------------------------------------------------------------------------")

    def print_model_info(self):
        if self.log_model_info:
            print("\n")
            print("MODEL INFO:")
            print(self.model_info)
            print("\n")

    def log_fitness(self, fitness):
        if self.wandb_enabled:
            if self.fitness_measure_name is not None:
                wandb.run.summary[self.fitness_measure_name] = fitness
            else:
                wandb.run.summary["fitness"] = fitness

    def log_fitness_std(self, fitness_std):
        if self.wandb_enabled:
            if self.fitness_measure_name is not None:
                wandb.run.summary[self.fitness_measure_name + "_std"] = fitness_std
            else:
                wandb.run.summary["fitness_std"] = fitness_std

    def calculate_layer_size(self, layer_shape):
        return functools.reduce(lambda x1, x2: x1 * x2, layer_shape[1:])

    def save_model(self, model):
        # model.save(filepath="temp_model.h5")
        # wandb.save("temp_model.h5")
        pass

    def _format_layer(self, current_layer: int):
        result = str(current_layer + 1)
        if len(result) == 1:
            result = f"0{current_layer + 1}"
        return result


class ConvModelInfoLogger(ModelInfoLogger):

    def __init__(self, wandb_enabled=False, wandb_project=None, log_keras_train=True, generation=None, child=None,
                 hyperparam_dict: dict = None, num_cols=None, log_model_info=True):
        super().__init__(wandb_enabled=wandb_enabled, wandb_project=wandb_project, log_keras_train=log_keras_train, generation=generation, child=child,
                         hyperparam_dict=hyperparam_dict, num_cols=num_cols, log_model_info=log_model_info)
        self.num_conv_layer = 0
        self.num_pool_layer = 0
        self.last_layer_was_pool = None

    def log_layer_info(self, current_layer, is_encoder_part):


        layer_number = self._format_layer(self.current_layer_index)
        if type(current_layer) == layers.Conv2D:
            current_layer: layers.Conv2D
            self.model_info += "class=<Conv2D>\t"
            self.model_info += f"size={current_layer.kernel_size}\t"
            if self.wandb_enabled and is_encoder_part:
                update_dict = {f"l{layer_number}_filters": current_layer.filters,
                               f"l{layer_number}_fsize": current_layer.kernel_size[0]}
                wandb.config.update(update_dict)
                self.num_conv_layer += 1

                if self.last_layer_was_pool is None:
                    self.last_layer_was_pool = False
                elif self.last_layer_was_pool == False:
                    layer_nr_pool = self._format_layer(self.current_layer_index - 1)
                    wandb.config.update({f"l{layer_nr_pool}_pool": False})
                else:
                    self.last_layer_was_pool = False

                self.layer_output_shapes[self.current_layer_index] = current_layer.output_shape
                self.current_layer_index += 1


        elif type(current_layer) == layers.MaxPool2D:
            self.model_info += "class=<MaxPool>\t"
            self.model_info += f"size={current_layer.pool_size}\t"
            if self.wandb_enabled and is_encoder_part:
                layer_number_pool = self._format_layer(self.current_layer_index - 1)
                wandb.config.update({f"l{layer_number_pool}_pool": True})
                self.num_pool_layer += 1
                self.last_layer_was_pool = True
                self.layer_output_shapes[self.current_layer_index - 1] = current_layer.output_shape

        elif type(current_layer) == layers.UpSampling2D:
            self.model_info += "class=<UpSampl>\t"
            self.model_info += f"size={current_layer.size}\t"

        elif type(current_layer) == layers.Dropout:
            self.model_info += "class=<Dropout>\t"
            self.model_info += f"rate={current_layer.rate}\t"

        self.model_info += f"output_shape={current_layer.output_shape} \n"

    def reached_end_encoder(self, last_layer):
        # in case last layer was not a pooling layer, add this to wandb here
        if self.wandb_enabled:
            if not self.last_layer_was_pool:
                layer_number_pool = self._format_layer(self.current_layer_index - 1)
                wandb.config.update({f"l{layer_number_pool}_pool": False})

            # log the layer sizes:
            min_layer_size = 10e5
            for layer, shape in self.layer_output_shapes.items():
                layer_number = self._format_layer(layer)
                layer_size = self.calculate_layer_size(shape)
                min_layer_size = min(min_layer_size, layer_size)
                wandb.config.update({f"l{layer_number}_size": layer_size})

            wandb.config.num_conv_layer = self.num_conv_layer
            wandb.config.num_pool_layer = self.num_pool_layer
            representation_layer_size = self.calculate_layer_size(last_layer.output_shape)
            wandb.config.update({"r_size": representation_layer_size})
            # As it is possible that the r_size is not the smallest size in the encoder, also save the true min.
            wandb.config.update({"min_lsize": min_layer_size})

            # Fill up everything until the last column
            for layer in range(self.current_layer_index, self.num_cols):
                layer_number = self._format_layer(layer)
                update_dict = {f"l{layer_number}_filters": "None",
                               f"l{layer_number}_fsize": "None",
                               f"l{layer_number}_pool": "None",
                               f"l{layer_number}_size": "None"}
                wandb.config.update(update_dict)

class ModelInfoLoggerFactory:

    def __init__(self, network_type, wandb_api_key=None, wandb_project=None, num_cols=None, log_keras_train=True,
                 log_model_info=True):
        self.network_type = network_type
        self.log_keras_train = log_keras_train
        self.log_model_info = log_model_info
        self.wandb_api_key = wandb_api_key
        self.run = None
        self.hyperparam_dict = {}
        self.num_cols = num_cols

        if wandb_api_key is not None:
            self.wandb_enabled = True
            assert wandb_project is not None, "If you want to enable wandb you also need to specify a project name."
            self.wandb_project = wandb_project
            wandb.login(key=self.wandb_api_key)
        else:
            self.wandb_enabled = False
            self.wandb_project = None

    def add_hyperparams(self, param_dict):
        """Specify all hyperparameters here that are the same for every individual of the evolutionary search."""
        if param_dict is not None:
            self.hyperparam_dict = {**self.hyperparam_dict, **param_dict}

    def create(self, generation=None, child=None):
        return ConvModelInfoLogger(generation=generation,
                                   child=child,
                                   wandb_enabled=self.wandb_enabled,
                                   wandb_project=self.wandb_project,
                                   log_keras_train=self.log_keras_train,
                                   log_model_info=self.log_model_info,
                                   hyperparam_dict=self.hyperparam_dict,
                                   num_cols=self.num_cols)


    def create_muted(self, log_keras_train = False, generation=None, child=None):
        return ConvModelInfoLogger(generation=generation,
                                   child=child,
                                   wandb_enabled=False,
                                   wandb_project=False,
                                   log_keras_train=log_keras_train,
                                   log_model_info=False,
                                   hyperparam_dict=self.hyperparam_dict,
                                   num_cols=self.num_cols)
