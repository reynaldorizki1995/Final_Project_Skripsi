# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Data Reza\Materi Kuliah\Skripsi\Sidang\Home.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from Testing import Ui_testsingWindow
from Training import Ui_trainingWindow

class Ui_Halaman_Utama(object):
    def openWindowTesting(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_testsingWindow()
        self.ui.setupUi(self.window)
        self.window.show()
    def openWindowTraining(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_trainingWindow()
        self.ui.setupUi(self.window)
        self.window.show()
    def setupUi(self, Halaman_Utama):
        Halaman_Utama.setObjectName("Halaman_Utama")
        Halaman_Utama.resize(619, 428)
        self.centralwidget = QtWidgets.QWidget(Halaman_Utama)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(110, 120, 421, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setScaledContents(False)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(200, 170, 251, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.testingButton = QtWidgets.QPushButton(self.centralwidget)
        self.testingButton.setGeometry(QtCore.QRect(200, 250, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.testingButton.setFont(font)
        self.testingButton.setObjectName("testingButton")
        self.testingButton.clicked.connect(self.openWindowTesting)
        self.trainingButton = QtWidgets.QPushButton(self.centralwidget)
        self.trainingButton.setGeometry(QtCore.QRect(360, 250, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.trainingButton.setFont(font)
        self.trainingButton.setObjectName("trainingButton")
        self.trainingButton.clicked.connect(self.openWindowTraining)
        Halaman_Utama.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Halaman_Utama)
        self.statusbar.setObjectName("statusbar")
        Halaman_Utama.setStatusBar(self.statusbar)

        self.retranslateUi(Halaman_Utama)
        QtCore.QMetaObject.connectSlotsByName(Halaman_Utama)

    def retranslateUi(self, Halaman_Utama):
        _translate = QtCore.QCoreApplication.translate
        Halaman_Utama.setWindowTitle(_translate("Halaman_Utama", "MainWindow"))
        self.label.setText(_translate("Halaman_Utama", "Pengenalan Ekspresi Wajah Menggunakan Metode "))
        self.label_2.setText(_translate("Halaman_Utama", "Convolutional Neural Network"))
        self.testingButton.setText(_translate("Halaman_Utama", "Testing"))
        self.trainingButton.setText(_translate("Halaman_Utama", "Training"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Halaman_Utama = QtWidgets.QMainWindow()
    ui = Ui_Halaman_Utama()
    ui.setupUi(Halaman_Utama)
    Halaman_Utama.show()
    sys.exit(app.exec_())
