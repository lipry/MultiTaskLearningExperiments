import yaml

with open('config/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
