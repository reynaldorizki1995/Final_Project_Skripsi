# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Data Reza\Materi Kuliah\Skripsi\Sidang\Testing.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import math
import cv2
import numpycnn
import os

# DATADIR = "D:/Data Reza/Materi Kuliah/Skripsi/Seminar/CitraTestingDatas/"
# CATEGORIES = ["Muak"]
# img_data_list = []
class Ui_testsingWindow(object):
    def setupUi(self, testsingWindow):
        testsingWindow.setObjectName("testsingWindow")
        testsingWindow.resize(458, 357)
        self.centralwidget = QtWidgets.QWidget(testsingWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.preprocessingButton = QtWidgets.QPushButton(self.centralwidget)
        self.preprocessingButton.setGeometry(QtCore.QRect(60, 80, 91, 31))
        self.preprocessingButton.setObjectName("preprocessingButton")
        self.preprocessingButton.clicked.connect(self.prepro)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(110, 40, 241, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pili_citra = QtWidgets.QPushButton(self.centralwidget)
        self.pili_citra.setGeometry(QtCore.QRect(180, 80, 91, 31))
        self.pili_citra.setObjectName("pili_citra")
        self.pili_citra.clicked.connect(self.openFile)
        self.testingButton = QtWidgets.QPushButton(self.centralwidget)
        self.testingButton.setGeometry(QtCore.QRect(290, 80, 91, 31))
        self.testingButton.setObjectName("testingButton")
        self.testingButton.clicked.connect(self.testing)
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(20, 180, 421, 151))
        self.textBrowser.setObjectName("textBrowser")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 150, 291, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        testsingWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(testsingWindow)
        self.statusbar.setObjectName("statusbar")
        testsingWindow.setStatusBar(self.statusbar)

        self.retranslateUi(testsingWindow)
        QtCore.QMetaObject.connectSlotsByName(testsingWindow)

    def openFile(self):
        global img
        fileName, _ = QFileDialog.getOpenFileName(None, "Pilih Citra", "", "Image Files (*.JPG)")
        if fileName:
            img = cv2.imread(fileName)
            cv2.imshow("Citra Asli", img)

    def prepro(self):
        global resize_img
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        smooth_img = cv2.GaussianBlur(gray_img, (3, 3), 1)
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
        ret, thresh = cv2.threshold(faceimg, 80, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(faceimg, contours, -1, (0, 255, 0), 6)
        pa = thresh.shape[0]
        la = thresh.shape[1]

        resize_img = np.zeros(shape=[222, 222], dtype=np.uint8)
        for i in range(0, pa):
            for j in range(0, la):
                if (thresh[i, j] == 255):
                    x = math.floor((222 * i) / pa)
                    y = math.floor((222 * j) / la)
                    resize_img[x, y] = thresh[i, j]
        cv2.imshow("resize",resize_img)
    def testing(self):

        filter_load = np.load("filter.npz")
        C1_filter = filter_load['f1']
        C2_filter = filter_load['f2']


        weight = np.load("bobot/ep_10/weight.npy")
        bias = np.load("bobot/ep_10/bias.npy")

        C1_feature_map = numpycnn.conv(resize_img,C1_filter)
        C1_feature_map_relu = numpycnn.relu(C1_feature_map)

        # S1 #
        C1_feature_map_relu_pool = numpycnn.pooling(C1_feature_map_relu)

        ## Layer 2 ##
        # C2 #
        C2_feature_map = numpycnn.conv(C1_feature_map_relu_pool, C2_filter)
        C2_feature_map_relu = numpycnn.relu(C2_feature_map)
        # S2 #
        C2_feature_map_relu_pool = numpycnn.pooling(C2_feature_map_relu)


        vektor = np.zeros((4, 1, 2916))
        for i in range(0, 4):
            vektor[i, :, :] = C2_feature_map_relu_pool[:, :, i].flatten()

        flatten = np.reshape(vektor, 11664).flatten()
        flatten = flatten.reshape((-1, 1))

        ### Fully-connected ###
        kelas = ["Bahagia", "Marah", "Muak", "Netral", "Sedih", "Takut", "Terkejut"]

        fully = []
        for i in range(0, len(kelas)):
            fully.append(np.sum(flatten * weight[i, :] + bias[i]))

        ## Exponensial ##
        eksponen = []
        for i in range(0, len(kelas)):
            eksponen.append(math.exp(fully[i]))

        ## Softmax ##
        softmax = []
        for i in range(0, len(kelas)):
            softmax.append(eksponen[i] / sum(eksponen))
        print("\n")
        print("--> Hasil prediksi :", softmax)
        print("Maka, ekspresi wajah yang terprediksi adalah :", kelas[np.argmax(softmax)])
        # self.textBrowser.append(softmax,kelas[np.argmax(softmax)])
    def retranslateUi(self, testsingWindow):
        _translate = QtCore.QCoreApplication.translate
        testsingWindow.setWindowTitle(_translate("testsingWindow", "MainWindow"))
        self.preprocessingButton.setText(_translate("testsingWindow", "Preprocessing"))
        self.label.setText(_translate("testsingWindow", "Pengenalan Ekspresi Wajah"))
        self.pili_citra.setText(_translate("testsingWindow", "Pilih Citra"))
        self.testingButton.setText(_translate("testsingWindow", "Testing"))
        self.label_2.setText(_translate("testsingWindow", "Hasil Prediksi Ekpresi Wajah Adalah:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    testsingWindow = QtWidgets.QMainWindow()
    ui = Ui_testsingWindow()
    ui.setupUi(testsingWindow)
    testsingWindow.show()
    sys.exit(app.exec_())
