import os

import numpy as np

from wandb.keras import WandbCallback

from autoencoder_cgp.evolutionary_components.configurations import EvolutionConfiguration, SearchConfiguration
from autoencoder_cgp.evolutionary_components.custom_callbacks import ReduceLRonPerformanceDrop, WandbSaveModelCallback, \
    CustomEarlyStopping
from autoencoder_cgp.evolutionary_components.individual import Individual

from autoencoder_cgp.evolutionary_components.logger import ModelInfoLoggerFactory
from autoencoder_cgp.evolutionary_components.network_block_table import NetworkBlockTable
from autoencoder_cgp.evolutionary_components.operations import initialize_individual_default, \
    generate_model_from_phenotype, mutate_passive_gene, mutate_active_gene_forced, random_sample_block_default

import autoencoder_cgp.evolutionary_components.constants as constants


def evolutionary_search(X_train, fitness_eval_func, fitness_func_args, maximize_fitness,
                        search_config: "SearchConfiguration", num_generations, mutation_probability, num_children,
                        num_rows, num_cols, levelback, path=None, start_generation=0,
                        start_individual: "Individual" = None, fitness_history=None, best_fitness_history=None,
                        generation_history=None, logger_factory: ModelInfoLoggerFactory = None,
                        input_layer_shape=None, num_evaluations=1, initialize_individual=initialize_individual_default,
                        random_sample_block_function=random_sample_block_default
                        ):
    epochs = search_config.epochs
    batch_size = search_config.batch_size

    # lists to track the results
    fitness_history = [] if fitness_history is None else fitness_history
    best_fitness_history = [] if best_fitness_history is None else best_fitness_history
    generation_history = [] if generation_history is None else generation_history

    # Initialization:
    network_table = NetworkBlockTable(search_config.block_search_space, network_type=search_config.network_type)
    if start_individual is None:
        model_logger = logger_factory.create(0, "parent")
        parent = initialize_individual(network_table, num_rows=num_rows, num_columns=num_cols, levelback=levelback,
                                       input_layer_shape=input_layer_shape,
                                       max_representation_size=search_config.max_representation_size)
        fitness_list = []
        for i in range(num_evaluations):
            print(f"\n \n Training the first individual {i + 1}/{num_evaluations}")
            model_logger_temp = logger_factory.create_muted(log_keras_train=True)
            parent.model = train_individual(parent, X_train, batch_size, epochs,
                                            input_layer_shape=input_layer_shape,
                                            model_info_logger=model_logger_temp if i > 0 else model_logger,
                                            network_type=search_config.network_type)
            fitness_temp = fitness_eval_func(parent.model, **fitness_func_args)
            fitness_list.append(fitness_temp)
        fitness_array = np.array(fitness_list)
        parent.fitness = fitness_array.mean()

        model_logger.log_fitness(parent.fitness)
        if num_evaluations > 1:
            model_logger.log_fitness_std(fitness_array.std())

        parent.generation = 0
        generation_history.append(0)
        fitness_history.append(parent.fitness)
        print(f"\n First individual scored a fitness of {parent.fitness} ({fitness_array.std()})")
    else:
        parent = start_individual
        parent.fitness = start_individual.fitness
        parent.model = start_individual.model

    # Begin of evolution:
    generation = start_generation
    while generation < num_generations + start_generation:

        print(f"Creating generation: {generation + 1}/{num_generations + start_generation}")
        child_fitnesses = []
        children = []
        for i in range(num_children):
            print(f"\n\nCreating and training child {i + 1}/{num_children}")
            model_logger = logger_factory.create(generation, child=i)
            child = mutate_active_gene_forced(parent, mutate_proba=mutation_probability, num_rows=num_rows,
                                              num_cols=num_cols, levelback=levelback, network_block_table=network_table,
                                              input_layer_shape=input_layer_shape,
                                              max_representation_size=search_config.max_representation_size,
                                              random_sample_block_function=random_sample_block_function)
            child.model = train_individual(child, X_train, batch_size, epochs,
                                           input_layer_shape=input_layer_shape,
                                           model_info_logger=model_logger,
                                           network_type=search_config.network_type)
            child.fitness = fitness_eval_func(child.model, **fitness_func_args)

            fitness_list = [child.fitness]
            # If child fitness is better retrain model and average fitness to make sure that this better fitness
            # did not only occur for change. This is only done if num_evaluations > 1.
            if child.fitness > parent.fitness and maximize_fitness \
                    or child.fitness < parent.fitness and not maximize_fitness:

                if num_evaluations > 1:
                    print("\n\nChild has new best fitness. Checking for variance of architecture by retraining. \n")
                for j in range(num_evaluations - 1):
                    print(f"Retraining {j + 1}/{num_evaluations - 1}")
                    model_logger_temp = logger_factory.create_muted(log_keras_train=True)
                    child.model = train_individual(child, X_train, batch_size, epochs,
                                                   input_layer_shape=input_layer_shape,
                                                   model_info_logger=model_logger_temp,
                                                   network_type=search_config.network_type)
                    fitness_temp = fitness_eval_func(child.model, **fitness_func_args)
                    fitness_list.append(fitness_temp)
            fitness_array = np.array(fitness_list)
            child.fitness = fitness_array.mean()

            model_logger.log_fitness_std(fitness_array.std())
            model_logger.log_fitness(child.fitness)
            child.generation = generation
            children.append(child)
            child_fitnesses.append(child.fitness)
            print(f"Child scored a fitness of {child.fitness}({fitness_array.std()})")

        if maximize_fitness:
            best_child_index = np.argmax(child_fitnesses)
        else:
            best_child_index = np.argmin(child_fitnesses)

        if child_fitnesses[best_child_index] > parent.fitness and maximize_fitness \
                or child_fitnesses[best_child_index] < parent.fitness and not maximize_fitness:
            parent = children[best_child_index]
            best_fitness = child_fitnesses[best_child_index]
        else:
            mutate_passive_gene(individual=parent, mutate_proba=mutation_probability, num_rows=num_rows,
                                num_cols=num_cols, levelback=levelback, network_block_table=network_table)
            best_fitness = parent.fitness
            parent.generation = generation

        __track_results(child_fitnesses, fitness_history, best_fitness, best_fitness_history, generation,
                        generation_history, num_children, parent, path)

        print(
            f"Evaluation of generation {generation + 1}/{num_generations + start_generation} finished. Best fitness: {best_fitness}")
        generation += 1
    history = {"generation": np.array(generation_history), "fitness": np.array(fitness_history),
               "best_fitness": np.array(best_fitness_history)}
    return parent, history


def __track_results(child_fitnesses, fitness_history, best_fitness, best_fitness_history, generation,
                    generation_history, num_children, parent: Individual,
                    path):
    generation_history += [generation] * num_children
    fitness_history += child_fitnesses
    best_fitness_history.append(best_fitness)
    if path is not None:
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            filename_individual = f"best_individual_generation_{generation}_fitness_{str(np.round(parent.fitness, 2))}.pickle"
            # In case you want to save the best model at each generation specify the following variable:
            filename_model = None  # f"best_model_generation_{generation}_fitness_{str(np.round(parent.fitness, 2))}.h5"
            parent.save(path, filename_individual, filename_model)
            evol_config = EvolutionConfiguration(generation_history=generation_history, fitness_history=fitness_history,
                                                 best_fitness_history=best_fitness_history,
                                                 filename_individual=filename_individual, filename_model=filename_model)
            evol_config.save(f"{path}/{constants.EVOL_CONFIG_NAME}")
        except Exception as e:
            print(f"Exception occurred. It was not possible to save individual to the path: {path} \n {e}")
            raise e
    return fitness_history, generation_history


def train_individual(individual, X_train, batch_size, epochs, input_layer_shape, model_info_logger,
                     network_type="Conv"):
    model = generate_model_from_phenotype(individual.phenotype, input_layer_shape, model_info_logger=model_info_logger)
    model_callbacks = initalize_callbacks(model_info_logger, network_type)

    model.fit(X_train, X_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=model_info_logger.keras_verbose,
              callbacks=model_callbacks)
    return model


def initalize_callbacks(model_info_logger, network_type="Conv"):
    model_callbacks = []
    if model_info_logger.wandb_enabled:
        model_callbacks.append(WandbCallback())
        save_model_callback = WandbSaveModelCallback(model_info_logger)
        model_callbacks.append(save_model_callback)

    early_stopping_callback = CustomEarlyStopping(monitor="loss", mode="min", baseline=50.0, patience=10,
                                                  restore_best_weights=True,
                                                  verbose=1,
                                                  min_delta=0.0001
                                                  )
    model_callbacks.append(early_stopping_callback)

    return model_callbacks


def initalize_callbacks_test():
    model_callbacks = []
    reduce_lr_on_performance_drop = ReduceLRonPerformanceDrop(monitor="loss",
                                                              patience=10,
                                                              min_delta=0.0001,
                                                              factor=0.5,
                                                              min_lr=10e-6,
                                                              mode="min",
                                                              verbose=1)
    model_callbacks.append(reduce_lr_on_performance_drop)
    return model_callbacks
