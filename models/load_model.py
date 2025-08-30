from models.intracross_model import IntraCross

def load_model(config, device):
    if config['MODEL'] == 'intracross':
        model = IntraCross(config).to(device)
    else:
        print('Model not recognised')
    return model