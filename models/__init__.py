network_names = {}


def register_network(name):
    def f(network):
        if name in network_names:
            print(f'Error when registering name for class {network.__class__.__name__}')
            raise Exception(f'Network name "{name}" is registered with class {network_names[name].__class__.__name__}')
        network_names[name] = network
        return network

    return f


from models.alexnet import *
from models.mobilenet import *
from models.resnet101 import *
from models.resnet50 import *
from models.mobilenetv3 import *
from models.mobilenetv3_L import *