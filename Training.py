# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Data Reza\Materi Kuliah\Skripsi\Sidang\Training.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import numpycnn
import cv2
import os
import random
import time
import math
import numpy as np

# CATEGORIES = ["Bahagia"]



class Ui_trainingWindow(object):
    def setupUi(self, trainingWindow):
        trainingWindow.setObjectName("trainingWindow")
        trainingWindow.resize(521, 370)
        self.centralwidget = QtWidgets.QWidget(trainingWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(10, 120, 501, 241))
        self.textBrowser.setObjectName("textBrowser")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(110, 10, 311, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.trainingButton = QtWidgets.QPushButton(self.centralwidget)
        self.trainingButton.setGeometry(QtCore.QRect(220, 60, 91, 41))
        self.trainingButton.setObjectName("trainingButton")
        self.trainingButton.clicked.connect(self.trainingCNN)
        trainingWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(trainingWindow)
        self.statusbar.setObjectName("statusbar")
        trainingWindow.setStatusBar(self.statusbar)

        self.retranslateUi(trainingWindow)
        QtCore.QMetaObject.connectSlotsByName(trainingWindow)

    def grayscale(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img_gray
    def smoothing(self, img_gray):
        smooth_img = cv2.GaussianBlur(img_gray, (3, 3), 1)
        return smooth_img
    def segmentasi(self, smooth_img):
        face_cascade = cv2.CascadeClassifier('D:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')
        faces = face_cascade.detectMultiScale(smooth_img, 1.3, 5)
        if len(faces) == 0:
            return
        for (column, row, width, height) in faces:
            smooth_img[row:row + width, column:column + height]
        for (column, row, width, height) in faces:
            r = max(width, height) / 2
            center_column = column + width / 2
            center_row = row + height / 2
            nx = int(center_column - r)
            ny = int(center_row - r)
            nr = int(r * 2)
            faceimg = smooth_img[ny:ny + nr, nx:nx + nr]
        return faceimg
    def threshold(self, faceimg):
        ret, thresh = cv2.threshold(faceimg, 80, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(faceimg, contours, -1, (0, 255, 0), 6)
        return thresh
    def resize(self, thresh):
        pa = thresh.shape[0]
        la = thresh.shape[1]

        resize_img = np.zeros(shape=[222, 222], dtype=np.uint8)
        for i in range(0, pa):
            for j in range(0, la):
                if (thresh[i, j] == 255):
                    x = math.floor((222 * i) / pa)
                    y = math.floor((222 * j) / la)
                    resize_img[x, y] = thresh[i, j]
        return resize_img



    def trainingCNN(self):
        start = time.time()
        DATADIR = "D:/Data Reza/Materi Kuliah/Skripsi/Seminar/CitraLatihTest/"
        CATEGORIES = ["Bahagia", "Marah", "Muak", "Netral", "Sedih", "Takut", "Terkejut"]
        img_data_list = []
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category) #Ambil folder
            class_num = CATEGORIES.index(category)
            for gambar in os.listdir(path):
                img = cv2.imread(os.path.join(path,gambar))
                gray = self.grayscale(img)
                smooth = self.smoothing(gray)
                segmen = self.segmentasi(smooth)
                thres = self.threshold(segmen)
                resize = self.resize(thres)

                img_data_list.append([resize, class_num])
                random.shuffle(img_data_list)

        X = []
        y = []

        for features, label in img_data_list:
            X.append(features)
            y.append(label)

        for i in range(0, len(y)):
            ohl = [0, 0, 0, 0, 0, 0, 0]
            ohl[y[i]] = 1
            y[i] = ohl

        y = np.array(y)
        weight = np.zeros((7, 11664, 1))
        for i in range(0, 6):
            for j in range(0, 11663):
                nilai = round(random.uniform(-0.5, 0.5), 6)
                weight[i, j, :] = nilai
        print("======Training CNN======")
        x=0
        bias = [0, 0, 0, 0, 0, 0, 0]
        epoch = 25
        lrate = 0.001
        # flatten_simpan = []
        # for ep in range(0, epoch):
        for ep in range(0, epoch):
            for image_i in range(0, len(X)):

                cetak = "Input Ke-" + str(image_i + 1)
                print(cetak)

                C1_filter = np.zeros((2, 3, 3))
                C1_filter[0, :, :] = np.array([[[-0.05, 0.02, 0.01],
                                                [0.04, -0.02, 0.05],
                                                [0.04, -0.01, 0.02]]])
                C1_filter[1, :, :] = np.array([[[0.04, 0.01, -0.03],
                                                [0.05, 0.02, 0],
                                                [-0.04, -0.01, 0.05]]])

                C1_feature_map = numpycnn.conv(X[image_i], C1_filter)

                C1_feature_map_relu = numpycnn.relu(C1_feature_map)

                C1_feature_map_relu_pool = numpycnn.pooling(C1_feature_map_relu)

                C2_filter = np.zeros((4, 3, 3, C1_feature_map_relu_pool.shape[-1]))
                for i in range(0, C1_feature_map_relu_pool.shape[-1]):
                    C2_filter[0, :, :, i] = np.array([[[-0.02, -0.02, -0.02],
                                                       [0.03, -0.05, 0.04],
                                                       [-0.04, 0.03, 0.03]
                                                       ]])

                    C2_filter[1, :, :, i] = np.array([[[-0.04, -0.04, 0.04],
                                                       [0.04, 0.02, 0.03],
                                                       [0.02, -0.03, 0.04]]])

                    C2_filter[2, :, :, i] = np.array([[[0.03, 0, 0.03],
                                                       [0.02, -0.04, 0.05],
                                                       [0.02, 0.05, -0.05]]])

                    C2_filter[3, :, :, i] = np.array([[[0.03, 0.04, -0.01],
                                                       [0.03, -0.05, 0.02],
                                                       [-0.04, 0.03, -0.03]]])

                C2_feature_map = numpycnn.conv(C1_feature_map_relu_pool, C2_filter)

                C2_feature_map_relu = numpycnn.relu(C2_feature_map)

                # S2
                C2_feature_map_relu_pool = numpycnn.pooling(C2_feature_map_relu)
                vektor = np.zeros((4, 1, 2916))
                for i in range(0, 4):
                    vektor[i, :, :] = C2_feature_map_relu_pool[:, :, i].flatten()
                flatten = np.reshape(vektor, 11664).flatten()
                flatten = flatten.reshape((-1, 1))

                kelas = 7

                fully = []
                for i in range(0, kelas):
                    fully.append(np.sum(flatten[image_i] * weight[i, :] + bias[i]))

                eksponen = []
                for i in range(0, kelas):
                    eksponen.append(math.exp(fully[i]))

                softmax = []
                for i in range(0, kelas):
                    softmax.append(eksponen[i] / sum(eksponen))

                ## Cross-entropy Loss ##
                # ohl = [1, 0, 0, 0, 0, 0, 0]
                ohl = y[image_i]
                loss = []
                terror = 0.00001

                for i in range(0, kelas):
                    loss.append(ohl[i] * math.log10(softmax[i]))

                loss = sum(loss) * -1

                if loss > terror:
                    cetak1 = "epoch ke"+str(ep +1)
                    print(cetak1)
                    delta_y = []
                    for i in range(0, kelas):
                        delta_y.append(softmax[i] - ohl[i])

                    delta_w = np.zeros((kelas, flatten[image_i].shape[0]))
                    for i in range(0, kelas):
                        for j in range(0, flatten[image_i].shape[0]):
                            delta_w[i, j] = delta_y[i] * flatten[image_i][j]

                    delta_b = delta_y

                    weight_baru = np.zeros_like(weight)
                    for i in range(0, kelas):
                        for j in range(0, weight.shape[1]):
                            weight_baru[i, j] = weight[i, j] - (lrate * delta_w[i, j])

                    weight = weight_baru

                    bias_baru = bias
                    for i in range(0, kelas):
                        bias_baru[i] = bias[i] - (lrate * delta_b[i])
                    bias = bias_baru

                    fully = []
                    for i in range(0, kelas):
                        fully.append(np.sum(flatten[image_i] * weight[i, :]) + bias[i])

                        ## Exponensial ##
                    eksponen = []
                    for i in range(0, kelas):
                        eksponen.append(math.exp(fully[i]))

                        ## Softmax ##
                    softmax = []
                    for i in range(0, kelas):
                        softmax.append(eksponen[i] / sum(eksponen))

                    cetak = "--> Softmax" + str(softmax)
                    print(cetak)
                    self.textBrowser.append(cetak)

            if (ep+1 == 10):
                np.save("bobot/ep_10/weight.npy", weight)
                np.save("bobot/ep_10/bias.npy", bias)
            elif (ep + 1 == 15):
                np.save("bobot/ep_15/weight.npy", weight)
                np.save("bobot/ep_15/bias.npy", bias)
            elif (ep + 1 == 20):
                np.save("bobot/ep_20/weight.npy", weight)
                np.save("bobot/ep_20/bias.npy", bias)
            elif (ep + 1 == 25):
                np.save("bobot/ep_25/weight.npy", weight)
                np.save("bobot/ep_25/bias.npy", bias)
        # elif (ep+1 == 100):
        #     np.save("bobot 2 layer/ep_100/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_100/bias.npy", bias)
        # elif (ep+1 == 150):
        #     np.save("bobot 2 layer/ep_150/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_150/bias.npy", bias)
        # elif (ep+1 == 200):
        #     np.save("bobot 2 layer/ep_200/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_200/bias.npy", bias)
        # elif (ep + 1 == 250):
        #     np.save("bobot 2 layer/ep_250/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_250/bias.npy", bias)
        # elif (ep+1 == 300):
        #     np.save("bobot 2 layer/ep_300/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_300/bias.npy", bias)
        # elif (ep+1 == 350):
        #     np.save("bobot 2 layer/ep_350/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_350/bias.npy", bias)
        # elif (ep+1 == 400):
        #     np.save("bobot 2 layer/ep_400/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_400/bias.npy", bias)
        # elif (ep+1 == 450):
        #     np.save("bobot 2 layer/ep_450/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_450/bias.npy", bias)
        # elif (ep+1 == 500):
        #     np.save("bobot 2 layer/ep_500/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_500/bias.npy", bias)
        # elif (ep+1 == 550):
        #     np.save("bobot 2 layer/ep_550/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_550/bias.npy", bias)
        # elif (ep+1 == 600):
        #     np.save("bobot 2 layer/ep_600/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_600/bias.npy", bias)
        # elif (ep+1 == 650):
        #     np.save("bobot 2 layer/ep_650/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_650/bias.npy", bias)
        # elif (ep+1 == 700):
        #     np.save("bobot 2 layer/ep_700/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_700/bias.npy", bias)
        # elif (ep+1 == 750):
        #     np.save("bobot 2 layer/ep_750/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_750/bias.npy", bias)
        # elif (ep+1 == 800):
        #     np.save("bobot 2 layer/ep_800/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_800/bias.npy", bias)
        # elif (ep+1 == 850):
        #     np.save("bobot 2 layer/ep_850/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_850/bias.npy", bias)
        # elif (ep+1 == 900):
        #     np.save("bobot 2 layer/ep_900/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_900/bias.npy", bias)
        # elif (ep+1 == 950):
        #     np.save("bobot 2 layer/ep_950/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_950/bias.npy", bias)
        # elif (ep+1 == 1000):
        #     np.save("bobot 2 layer/ep_1000/weight.npy", weight)
        #     np.save("bobot 2 layer/ep_1000/bias.npy", bias)

        # np.savez("test_epoch_bobot14_Lrate_0,01_epoch_100/filter.npz", f1=C1_filter, f2=C2_filter, f3=C3_filter, f4=C4_filter, f5=C5_filter)
        # np.save("test_epoch_bobot14_Lrate_0,01_epoch_100/weight.npy", weight)
        # np.save("test_epoch_bobot14_Lrate_0,01_epoch_100/bias.npy", bias)

        # print("Selesai")
        end = time.time()
        cetak = "Lamanya Proses : " + str(int(round(end - start))) + "s"
        print(cetak)

    def retranslateUi(self, trainingWindow):
        _translate = QtCore.QCoreApplication.translate
        trainingWindow.setWindowTitle(_translate("trainingWindow", "MainWindow"))
        self.label.setText(_translate("trainingWindow", "Training Pengenalan Ekspresi Wajah"))
        self.trainingButton.setText(_translate("trainingWindow", "Mulai Training"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    trainingWindow = QtWidgets.QMainWindow()
    ui = Ui_trainingWindow()
    ui.setupUi(trainingWindow)
    trainingWindow.show()
    sys.exit(app.exec_())
