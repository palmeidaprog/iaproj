from typing import List

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, Layer
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model

from data import Data
from metrics import Metrics


class CNN:

    def __init__(self, data: Data):
        self.data = data

    def __dense_dropout(self, layers: Layer, units: int, dropout_rate: float) -> Layer:
        layers = Dense(units=units, activation='relu')(layers)
        return Dropout(rate=dropout_rate)(layers)

    """
    Cria camada intermediaria convolucional padrão
    """

    def __create_hidden_layer(self, layers: Layer, kernels: int, dropout: bool = False) -> Layer:
        layers = SeparableConv2D(filters=kernels, kernel_size=(3, 3), activation='relu',
                                 padding='same')(layers)
        layers = SeparableConv2D(filters=kernels, kernel_size=(3, 3), activation='relu',
                                 padding='same')(layers)
        layers = BatchNormalization()(layers)
        layers = MaxPool2D(pool_size=(2, 2))(layers)

        if (dropout):
            return Dropout(rate=0.2)(layers)
        else:
            return layers

    def __input_layer(self, keras_input: Input) -> Layer:
        layers = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(
            keras_input)
        layers = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(layers)
        return MaxPool2D(pool_size=(2, 2))(layers)

    def __hidden_layers(self, layers: Layer) -> Layer:
        layers = self.__create_hidden_layer(layers, 32)  # 2a camada
        layers = self.__create_hidden_layer(layers, 64)  # 3a camada
        layers = self.__create_hidden_layer(layers, 128, True)  # 4a camada
        return self.__create_hidden_layer(layers, 256, True)  # 5a camada

    def __connected_layers(self, layers: Layer) -> Layer:
        layers = Flatten()(layers)
        layers = self.__dense_dropout(layers, 512, 0.7)
        layers = self.__dense_dropout(layers, 128, 0.5)
        return self.__dense_dropout(layers, 64, 0.3)

    def train(self, epochs: int = 10) -> None:
        self.epochs = epochs
        metrics = ['accuracy']
        keras_input = Input(shape=(self.data.image_dimension, self.data.image_dimension, 3))

        # Camadas
        layers: Layer = self.__input_layer(keras_input)
        layers = self.__hidden_layers(layers)
        layers = self.__connected_layers(layers)
        prediction_layer: Layer = Dense(units=1, activation='sigmoid')(layers)

        # Modelo
        self.model = Model(inputs=keras_input, outputs=prediction_layer)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
        step_size = self.data.train_generated_images.samples // self.data.batch_size
        validation_steps = self.data.test_validation_generated_images.samples // self.data.batch_size
        self.histogram = self.model.fit(self.data.train_generated_images,
                                        steps_per_epoch=step_size, epochs=self.epochs,
                                        validation_data=self.data.train_generated_images,
                                        validation_steps=validation_steps,
                                        callbacks=self.__callbacks())

    def __callbacks(self) -> List[Callback]:
        checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True,
                                     save_weights_only=True)
        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2,
                                   mode='max')
        return [checkpoint, reduce]

    def test(self) -> Metrics:
        return Metrics(self.model.predict(self.data.test_images), self.data, self.histogram)
