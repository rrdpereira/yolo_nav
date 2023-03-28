# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'controller_window.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(720, 560)
        self.graphicsView_2 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_2.setGeometry(QtCore.QRect(5, 5,1, 1))
        self.graphicsView_2.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(130, 650, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(280, 7, 41, 17))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.navigate_button = QtWidgets.QPushButton(Form)
        self.navigate_button.setGeometry(QtCore.QRect(250, 490, 231, 31))
        self.navigate_button.setObjectName("navigate_button")
        self.camera_slider = QtWidgets.QSlider(Form)
        self.camera_slider.setGeometry(QtCore.QRect(5, 5, 1, 1))
        self.camera_slider.setMinimum(-180)
        self.camera_slider.setMaximum(180)
        self.camera_slider.setOrientation(QtCore.Qt.Horizontal)
        self.camera_slider.setObjectName("camera_slider")
        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(30, 5, 650, 490))
        self.graphicsView.setObjectName("graphicsView")
        self.camera_image = QtWidgets.QLabel(Form)
        self.camera_image.setGeometry(QtCore.QRect(920, 8, 141, 17))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.camera_image.setFont(font)
        self.camera_image.setObjectName("camera_image")

        self.retranslateUi(Form)
        self.navigate_button.clicked.connect(Form.navigate)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "controller"))
        self.label.setText(_translate("Form", ""))
        self.label_2.setText(_translate("Form", ""))
        self.navigate_button.setText(_translate("Form", "navigate to the nearest object"))
        self.camera_image.setText(_translate("Form", ""))
