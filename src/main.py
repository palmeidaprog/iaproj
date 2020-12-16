import numpy as np
import tensorflow as tf

from cnn import CNN
from data import Data


def enable_gpu_legacy() -> None:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "GPU não encontrada!"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

def run(batch: int, idim: int, epochs: int) -> None:
    data = Data(batch, idim, epochs, '../dataset/2nd_split/')
    cnn = CNN(data)
    cnn.train(epochs)
    cnn.test()

def grid() -> None:
    for batch in [32, 64, 128, 256]:
        for idim in [150, 300]:
            for epochs in [5, 10, 15, 20, 25, 30]:
                run(batch, idim, epochs)

def main() -> None:
    enable_gpu_legacy()
    set_seed(1)

    #grid()
    run(32, 150, 30)

main()
print('Finalizado!')