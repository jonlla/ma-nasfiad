import autoencoder_cgp


def load_architecture_from_file(path, input_layer_shape, auto_compile_model=True, ):
    individual = autoencoder_cgp.evolutionary_components.individual.Individual.load_individual(path)
    model = autoencoder_cgp.evolutionary_components.operations.generate_model_from_phenotype(individual.phenotype,
                                                                                             input_layer_shape=input_layer_shape,
                                                                                             compile_model=auto_compile_model)
    return model
