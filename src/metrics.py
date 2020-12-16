import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from pandas import np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from mysql import connector

from data import Data


class Metrics:

    def __init__(self, predict: np.array, data: Data, hist: any):
        self.predict = predict
        self.data = data
        self.history = hist
        predict_rounded = np.round(self.predict)
        self.accuracy = accuracy_score(self.data.test_labels, predict_rounded) * 100
        self.cm = confusion_matrix(self.data.test_labels, predict_rounded)
        self.f1_score = metrics.f1_score(self.data.test_labels, predict_rounded) * 100
        self.recall = metrics.recall_score(self.data.test_labels, predict_rounded) * 100
        self.precision = metrics.precision_score(self.data.test_labels, predict_rounded) * 100
        self.mcc = metrics.matthews_corrcoef(self.data.test_labels, predict_rounded) * 100
        self.balanced_accuracy = metrics.balanced_accuracy_score(self.data.test_labels, predict_rounded) * 100
        self.__plot_confusion_matrix()
        self.save_to_db()

    def __plot_confusion_matrix(self) -> None:
        self.tn, self.fp, self.fn, self.tp = self.cm.ravel()

        self.__save_train_graph()
        plt.figure()
        plot_confusion_matrix(self.cm, figsize=(12, 8), hide_ticks=True)
        plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
        plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
        plt.savefig('cm.jpg')

        print('\n=== METRICAS ===\n')
        print(f'Accuracy: {self.accuracy}%')
        print(f'Precision: {self.precision}%')
        print(f'F1-score: {self.f1_score}%')
        print(f'Recall: {self.recall}%')
        print(f'MCC: {self.mcc}%')
        print(f'Balanced Accuracy: {self.balanced_accuracy}%')
        self.train_accuracy = np.round((self.history.history['accuracy'][-1]) * 100, 2)
        print(f'Train accuracy: {self.train_accuracy}')

    def save_to_db(self):
        cnx = connector.connect(user='iaproj', password='iaproj', host='localhost', database='ia')
        cursor = cnx.cursor()
        insert_sql = "insert into metrics(train_id, accuracy, trained_accuracy, precision_score, f1_score, recall, mcc, balanced_accuracy, true_negative, true_positive, false_negative, false_positive) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(insert_sql, (self.data.train_id, self.accuracy, self.train_accuracy, self.precision, self.f1_score, self.recall, self.mcc, self.balanced_accuracy, int(self.tn), int(self.tp), int(self.fn), int(self.fp)))
        cnx.commit()
        cursor.close()
        cnx.close()

    def __save_train_graph(self):
        figure, axis = plt.subplots(1, 2, figsize=(10, 3))
        axis = axis.ravel()

        for i, metrics in enumerate(['accuracy', 'loss']):
            axis[i].plot(self.history.history[metrics])
            axis[i].plot(self.history.history['val_' + metrics])
            axis[i].set_title(f'Model {metrics}')
            axis[i].set_xlabel('epochs')
            axis[i].set_ylabel(metrics)
            axis[i].legend(['train', 'val'])

        plt.savefig('treino.jpg')


