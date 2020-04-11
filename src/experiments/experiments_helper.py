from src.config.config import config


def get_batch_size():
    n_gpu = config['execution']['n_gpu']
    batch_size = config['cnn_full_params_sharing']['batch_size']
    return n_gpu * batch_size if n_gpu > 0 else batch_size


def check_input_type(admitted_input, message):
    if config['general']['input_type'] not in admitted_input:
        raise ValueError(message)