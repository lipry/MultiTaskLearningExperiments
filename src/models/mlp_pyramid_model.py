from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Concatenate, Dropout, Dense
from tensorflow.keras import Input, Model

from src.config.config import config
from src.metrics.metrics import auprc, auroc


def mlp_pyramid_model(combinations_dict, input_dims, hp):
    config_mlp = config['mlp_pyramidal']
    config_mlp_hp = config['mlp_pyramidal']['bayesian_opt']['hyperparameters']

    # Varible hyperparameters (choosen by bayesian optimizer)
    input_neurons_configuration = get_neurons_configuration(combinations_dict['input'], hp, 'input')
    main_neurons_configuration = get_neurons_configuration(combinations_dict['main'], hp, 'main')
    output_neurons_configuration = get_neurons_configuration(combinations_dict['output'], hp, 'output')

    print(input_neurons_configuration)
    print(main_neurons_configuration)
    print(output_neurons_configuration)

    learning_rate = hp.Float('learning_rate', min_value=config_mlp_hp['learning_rate'][0],
                             max_value=config_mlp_hp['learning_rate'][1])

    # Fixed hyper parameters
    # TODO: fixed droput rigth now.
    dropout = config_mlp['dropout']
    decay = config_mlp['decay']
    momentum = config_mlp['momentum']
    nesterov = config_mlp['nesterov']
    regularizer_lambda = config_mlp['regularizer_lambda'] if config_mlp['kernel_regularizer'] else 0.0



    inputs = [build_input_branch(c, dim, input_neurons_configuration, dropout, regularizer_lambda)
              for dim, c in zip(input_dims, config['general']['cell_lines'])]

    x = Concatenate()([i[1] for i in inputs])
    for neurons in main_neurons_configuration:
        x = Dense(neurons, activation="relu", kernel_regularizer=l2(regularizer_lambda))(x)
        x = Dropout(dropout)(x)

    outputs = [build_output_branch(c, output_neurons_configuration, dropout, regularizer_lambda, x) for c in config['general']['cell_lines']]

    mlp_model = Model([i[0] for i in inputs], outputs)

    sgd_opt = SGD(lr=learning_rate,
                  decay=decay,
                  momentum=momentum,
                  nesterov=nesterov)

    losses = {"pred_{}".format(c): 'binary_crossentropy' for c in config['general']['cell_lines']}
    mlp_model.compile(loss=losses,
                       optimizer=sgd_opt,
                       metrics=[auprc, auroc])

    return mlp_model

def get_hidden_layers_combinations(hidden_layers, max_level = 3, allow_empty_first_level=True):
    hiddenLayersList = []
    # First round manually...
    h1 = [[i] for i in hidden_layers[0]]
    hiddenLayersList.append(h1)
    # ...and from the second iteratively
    for i in range(1, max_level):
        tempList = []
        for k in hiddenLayersList[-1]:
            for j in hidden_layers[i]:
                if k[-1] > j:
                    tempitem = list(k)
                    tempitem.append(j)
                    tempList.append(tempitem)
        hiddenLayersList.append(tempList)
    # Add level [] if requested
    if allow_empty_first_level:
        hiddenLayersList.insert(0, [])
    # Sort the list according to the total number of neurons in the entire net
    tempLL = []
    for ll in hiddenLayersList:
        tempLL.append(sorted(ll, key=lambda x: sum(x)))

    return tempLL

def get_neurons_configuration(combinations, hp, conf):
    input_num_hidden_layers = hp.Int('{}_layers'.format(conf), min_value=0, max_value=len(combinations)-1)
    input_hidden_layer_choice = hp.Int('{}_neurons_comb'.format(conf), min_value=0, max_value=len(combinations[-1])-1)
    input_hidden_layer_choice = int(input_hidden_layer_choice * len(combinations[input_num_hidden_layers])
                              / len(combinations[-1]))
    return combinations[input_num_hidden_layers][input_hidden_layer_choice]

def get_combinations_dict():
    config_mlp_hp = config['mlp_pyramidal']['bayesian_opt']['hyperparameters']
    return {x: get_hidden_layers_combinations(config_mlp_hp['{}_neurons'.format(x)],
                                                             config_mlp_hp['{}_layers'.format(x)][1],
                                                             False) for x in ['input', 'output', 'main']}

def build_branch(neurons_conf, dropout, regularizer_lambda, prev):
    x = Dense(neurons_conf[0], activation="relu",  kernel_regularizer=l2(regularizer_lambda))(prev)
    x = Dropout(dropout)(x)
    for neurons in neurons_conf[1:]:
        x = Dense(neurons, activation="relu",  kernel_regularizer=l2(regularizer_lambda))(x)
        x = Dropout(dropout)(x)

    return x

def build_input_branch(branch_name, input_dim, neurons_conf, dropout, regularizer_lambda):
    input = Input(shape=(input_dim, ), name="input_{}".format(branch_name))
    x = build_branch(neurons_conf, dropout, regularizer_lambda, input)
    return input, x

def build_output_branch(branch_name, neurons_conf, dropout, regularizer_lambda, prev):
    x = build_branch(neurons_conf, dropout, regularizer_lambda, prev)
    pred = Dense(1, activation='sigmoid', name="pred_{}".format(branch_name))(x)
    return pred
