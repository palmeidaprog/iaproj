import os
from datetime import datetime
from os import path
from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator
from mysql import connector
from pandas import np

"""
Guarda dados, processa imagens para o treinamento e salva dados no banco
TODO: lembrar de remover hardcoded 'NORMAL', 'PNEUMONIA' do codigo
"""
class Data:

    def __init__(self, batch_size: int, image_dimension: int, epochs: int, folder: str = '../dataset',
                 split_folders: List[str] = ['train', 'val', 'test']):
        self.folder = folder
        self.split_folders = split_folders
        self.batch_size = batch_size
        self.epochs = epochs
        self.image_dimension = image_dimension
        self.__save_to_db()
        self.__inspect_data()
        self.__process_data()

    """
    Salva a quantidade de dados nos datasets
    """
    def __inspect_data(self) -> None:
        self.json: Dict[str, any] = {}

        for split_folder in self.split_folders:
            self.json[split_folder] = {}
            self.json[split_folder]['nomal_size'] =  len(os.listdir(path.join(self.folder, split_folder, 'NORMAL')))
            self.json[split_folder]['pneumonia_size'] = len(os.listdir(path.join(self.folder, split_folder, 'PNEUMONIA')))


    def __initialize_plot(self):
        self.fig, ax = plt.subplots(2, 3, figsize=(15, 7))
        self.ax = ax.ravel()
        plt.tight_layout()

    def __get_dataset_id(self, cursor: any) -> None:
        dataset_id_query = "select id from dataset where folder_name = %s"
        cursor.execute(dataset_id_query, (self.folder,))
        for (dataset_id, ) in cursor:
            self.dataset_id: int = dataset_id

    """
    Inicializa o cliente do mongodb
    """
    def __save_to_db(self) -> None:
        self.cnx = connector.connect(user='iaproj', password='iaproj', host='localhost', database='ia')
        cursor = self.cnx.cursor()
        self.__get_dataset_id(cursor)
        self.__insert_dataset(cursor)
        self.cnx.commit()
        cursor.close()
        self.cnx.close()
        print('Salvo no banco dados do dataset')


    def __insert_dataset(self, cursor):
        insert_sql = "insert into train(datetime, dataset_id, image_dimension, batch_size, epochs) " \
                     "values (%s, %s, %s, %s, %s)"
        cursor.execute(insert_sql, (datetime.now(), self.dataset_id, self.image_dimension, self.batch_size, self.epochs))
        self.train_id = cursor.lastrowid


    def __generate_images(self, dataFolder: str, generator: any) -> DirectoryIterator:
        return generator.flow_from_directory(directory=path.join(self.folder, dataFolder),
                                             target_size=(self.image_dimension, self.image_dimension),
                                             batch_size=self.batch_size, class_mode='binary', shuffle=True)

    """
    Gera as imagens a partir das imagens do dataset   
    """
    def __process_data(self) -> None:
        train_generator_data = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3,
                                                  vertical_flip=True)
        test_validation_generator_data = ImageDataGenerator(rescale=1. / 255)
        self.train_generated_images = self.__generate_images('train', train_generator_data )
        self.test_validation_generated_images = \
            self.__generate_images('test', test_validation_generator_data)
        float_type = 'float32'

        test_data = []
        test_labels = []
        test_folder = path.join(self.folder, 'test');

        for label, class_folder in enumerate(['NORMAL', 'PNEUMONIA']):
            for image in (os.listdir(path.join(test_folder, class_folder))):
                image = plt.imread(path.join(test_folder, class_folder, image))
                image = cv2.resize(image, (self.image_dimension, self.image_dimension))
                image = np.dstack([image, image, image])
                image = image.astype(float_type) / 255
                test_data.append(image)
                test_labels.append(label)

        self.test_images = np.array(test_data)
        self.test_labels = np.array(test_labels)