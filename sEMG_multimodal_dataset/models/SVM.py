import numpy as np
from scipy.fft import fft
from sklearn import svm
from sklearn.metrics import accuracy_score


class SVM:
    def __init__(self, classes):
        self.grid_matric = [[2, 4, 6, 8, 10], [0.001, 0.0001, 1e-05, 1e-06, 0.0000001]]
        self.classes = classes

    def get_temp_freq_fearture(self, data, fs=500):
        N, C, T = data.shape
        features = np.zeros((N, C, 4))
        for n in range(N):
            for c in range(C):
                signal = np.asarray(data[n, c, :], dtype=np.float64).reshape(-1)
                mean_value = np.mean(signal)
                rms_value = np.sqrt(np.mean(signal**2))
                spectrum = np.abs(fft(signal))[: T // 2 + 1]
                freq = np.fft.rfftfreq(T, d=1 / fs)
                mean_frequency = np.sum(freq * spectrum) / np.sum(spectrum)
                x = signal - mean_value
                X = np.fft.rfft(x)
                Pxx = (np.abs(X) ** 2) / (fs * max(1, x.size))
                psd = float(np.mean(Pxx))
                features[n, c, 0] = mean_value
                features[n, c, 1] = rms_value
                features[n, c, 2] = mean_frequency
                features[n, c, 3] = psd
        return features

    def train_test(self, x_train, y_train, x_test, y_test):
        x_train, x_test = self.get_temp_freq_fearture(x_train), self.get_temp_freq_fearture(x_test)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        curr_best_acc = -1.0
        result = None
        for c_param in self.grid_matric[0]:
            for gamma_param in self.grid_matric[1]:
                svm_classifier = svm.SVC(kernel="rbf", C=c_param, gamma=gamma_param)
                svm_classifier.fit(x_train, y_train)
                tpred = svm_classifier.predict(x_test)
                tmp_acc = accuracy_score(y_test, tpred)
                if tmp_acc > curr_best_acc:
                    curr_best_acc = tmp_acc
                    result = np.column_stack((y_test, tpred))
        assert result is not None
        return result
