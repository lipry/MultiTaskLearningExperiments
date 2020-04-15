from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Concatenate, Dropout, Dense
from tensorflow.keras import Input, Model

from src.config.config import config
from src.metrics.metrics import auprc, auroc


def mlp_model(input_dims, hp):
    config_mlp = config['multi_layers_perceptron']
    config_mlp_hp = config['multi_layers_perceptron']['bayesian_opt']['hyperparameters']

    # Varible hyperparameters (choosen by bayesian optimizer)
    input_layers = hp.Int('input_layers', min_value=config_mlp_hp['input_layers'][0],
                             max_value=config_mlp_hp['input_layers'][1])
    input_neurons = hp.Choice('input_neurons', values=config_mlp_hp['input_neurons'])
    body_layers = hp.Int('main_layers', min_value=config_mlp_hp['main_layers'][0],
                             max_value=config_mlp_hp['main_layers'][1])
    main_neurons = hp.Choice('main_neurons', values=config_mlp_hp['main_neurons'])
    output_layers = hp.Int('output_layers', min_value=config_mlp_hp['output_layers'][0],
                             max_value=config_mlp_hp['output_layers'][1])
    output_neurons = hp.Choice('output_neurons', values=config_mlp_hp['output_neurons'])

    learning_rate = hp.Float('learning_rate', min_value=config_mlp_hp['learning_rate'][0],
                             max_value=config_mlp_hp['learning_rate'][1])

    # Fixed hyper parameters
    # TODO: fixed droput rigth now.
    dropout = config_mlp['dropout']
    decay = config_mlp['decay']
    momentum = config_mlp['momentum']
    nesterov = config_mlp['nesterov']


    inputs = [build_input_branch(c, dim, input_layers, input_neurons, dropout)
              for dim, c in zip(input_dims, config['general']['cell_lines'])]

    x = Concatenate()([i[1] for i in inputs])
    for layer in range(body_layers):
        x = Dense(main_neurons, activation="relu")(x)
        x = Dropout(dropout)(x)

    outputs = [build_output_branch(c, output_layers, output_neurons, dropout, x) for c in config['general']['cell_lines']]

    mlp_model = Model([i[0] for i in inputs], outputs)

    sgd_opt = SGD(lr=learning_rate,
                  decay=decay,
                  momentum=momentum,
                  nesterov=nesterov)

    losses = {"pred_{}".format(c): 'binary_crossentropy' for c in config['general']['cell_lines']}
    mlp_model.compile(loss=losses,
                       optimizer=sgd_opt,
                       metrics=[auprc, auroc])

    print(mlp_model.summary())

    return mlp_model

def build_input_branch(branch_name, input_dim, n_layers, n_neurons, dropout):
    print(input_dim)
    input = Input(shape=(input_dim, ), name="input_{}".format(branch_name))

    x = Dense(n_neurons, activation="relu")(input)
    x = Dropout(dropout)(x)
    for layer in range(n_layers-1):
        x = Dense(n_neurons, activation="relu")(x)
        x = Dropout(dropout)(x)

    return input, x

def build_output_branch(branch_name, n_layers, n_neurons, dropout, prev):
    x = Dense(n_neurons, activation="relu")(prev)
    x = Dropout(dropout)(x)
    for layer in range(n_layers-1):
        x = Dense(n_neurons, activation="relu")(x)
        x = Dropout(dropout)(x)

    pred = Dense(1, activation='sigmoid', name="pred_{}".format(branch_name))(x)

    return pred
