# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'designer.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1133, 885)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_example = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab_example.sizePolicy().hasHeightForWidth())
        self.tab_example.setSizePolicy(sizePolicy)
        self.tab_example.setObjectName("tab_example")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_example)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.tab_example)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAutoFillBackground(True)
        self.label_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.tab_example)
        self.label.setMinimumSize(QtCore.QSize(101, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.sbCrop = QtWidgets.QSpinBox(self.tab_example)
        self.sbCrop.setMinimumSize(QtCore.QSize(60, 0))
        self.sbCrop.setMaximumSize(QtCore.QSize(85, 16777215))
        self.sbCrop.setMaximum(100)
        self.sbCrop.setSingleStep(10)
        self.sbCrop.setProperty("value", 100)
        self.sbCrop.setObjectName("sbCrop")
        self.horizontalLayout_3.addWidget(self.sbCrop)
        self.label_13 = QtWidgets.QLabel(self.tab_example)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_3.addWidget(self.label_13)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.label_11 = QtWidgets.QLabel(self.tab_example)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setAutoFillBackground(True)
        self.label_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_11.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_11.setObjectName("label_11")
        self.verticalLayout.addWidget(self.label_11)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.cbYellow = QtWidgets.QCheckBox(self.tab_example)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.cbYellow.setFont(font)
        self.cbYellow.setObjectName("cbYellow")
        self.horizontalLayout_9.addWidget(self.cbYellow)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.cbCircle = QtWidgets.QCheckBox(self.tab_example)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.cbCircle.setFont(font)
        self.cbCircle.setObjectName("cbCircle")
        self.verticalLayout.addWidget(self.cbCircle)
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.verticalLayout.addLayout(self.horizontalLayout_21)
        self.gridLayout_14 = QtWidgets.QGridLayout()
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.sbBlob = QtWidgets.QSpinBox(self.tab_example)
        self.sbBlob.setMinimumSize(QtCore.QSize(60, 0))
        self.sbBlob.setMaximumSize(QtCore.QSize(85, 16777215))
        self.sbBlob.setMaximum(200)
        self.sbBlob.setSingleStep(10)
        self.sbBlob.setProperty("value", 50)
        self.sbBlob.setObjectName("sbBlob")
        self.gridLayout_14.addWidget(self.sbBlob, 0, 2, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.tab_example)
        self.label_12.setObjectName("label_12")
        self.gridLayout_14.addWidget(self.label_12, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.tab_example)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_14.addWidget(self.label_2, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_14)
        self.label_15 = QtWidgets.QLabel(self.tab_example)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setAutoFillBackground(True)
        self.label_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_15.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_15.setObjectName("label_15")
        self.verticalLayout.addWidget(self.label_15)
        self.cbGreen = QtWidgets.QCheckBox(self.tab_example)
        self.cbGreen.setObjectName("cbGreen")
        self.verticalLayout.addWidget(self.cbGreen)
        self.cbMask = QtWidgets.QCheckBox(self.tab_example)
        self.cbMask.setObjectName("cbMask")
        self.verticalLayout.addWidget(self.cbMask)
        self.cbContours = QtWidgets.QCheckBox(self.tab_example)
        self.cbContours.setObjectName("cbContours")
        self.verticalLayout.addWidget(self.cbContours)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.pbExport = QtWidgets.QPushButton(self.tab_example)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.pbExport.setFont(font)
        self.pbExport.setObjectName("pbExport")
        self.horizontalLayout_6.addWidget(self.pbExport)
        self.pbUpdate = QtWidgets.QPushButton(self.tab_example)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.pbUpdate.setFont(font)
        self.pbUpdate.setObjectName("pbUpdate")
        self.horizontalLayout_6.addWidget(self.pbUpdate)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.gridLayout_2.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEditImage = QtWidgets.QLineEdit(self.tab_example)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.lineEditImage.setFont(font)
        self.lineEditImage.setObjectName("lineEditImage")
        self.horizontalLayout.addWidget(self.lineEditImage)
        self.pbImageOpen = QtWidgets.QPushButton(self.tab_example)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.pbImageOpen.setFont(font)
        self.pbImageOpen.setObjectName("pbImageOpen")
        self.horizontalLayout.addWidget(self.pbImageOpen)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.mpl = MplWidget(self.tab_example)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mpl.sizePolicy().hasHeightForWidth())
        self.mpl.setSizePolicy(sizePolicy)
        self.mpl.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.mpl.setObjectName("mpl")
        self.verticalLayout_3.addWidget(self.mpl)
        self.gridLayout_2.addLayout(self.verticalLayout_3, 1, 1, 1, 1)
        self.tabWidget.addTab(self.tab_example, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.layoutWidget = QtWidgets.QWidget(self.tab)
        self.layoutWidget.setGeometry(QtCore.QRect(170, 170, 810, 327))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_11.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.label_27 = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_27.sizePolicy().hasHeightForWidth())
        self.label_27.setSizePolicy(sizePolicy)
        self.label_27.setMaximumSize(QtCore.QSize(28, 16777215))
        self.label_27.setText("")
        self.label_27.setScaledContents(False)
        self.label_27.setWordWrap(True)
        self.label_27.setObjectName("label_27")
        self.gridLayout_11.addWidget(self.label_27, 11, 2, 1, 1)
        self.lineEditDirOutLeaf = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditDirOutLeaf.setObjectName("lineEditDirOutLeaf")
        self.gridLayout_11.addWidget(self.lineEditDirOutLeaf, 10, 2, 1, 1)
        self.pbDirectoryExport_3 = QtWidgets.QPushButton(self.layoutWidget)
        self.pbDirectoryExport_3.setObjectName("pbDirectoryExport_3")
        self.gridLayout_11.addWidget(self.pbDirectoryExport_3, 10, 5, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(77, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_11.addItem(spacerItem1, 2, 6, 1, 1)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.label_28 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_28.setFont(font)
        self.label_28.setAutoFillBackground(True)
        self.label_28.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_28.setObjectName("label_28")
        self.horizontalLayout_19.addWidget(self.label_28)
        self.gridLayout_11.addLayout(self.horizontalLayout_19, 0, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_11.addItem(spacerItem2, 1, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_11.addItem(spacerItem3, 0, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_11.addItem(spacerItem4, 2, 0, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_29.setFont(font)
        self.label_29.setAutoFillBackground(True)
        self.label_29.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_29.setObjectName("label_29")
        self.gridLayout_11.addWidget(self.label_29, 3, 2, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_11.addItem(spacerItem5, 3, 0, 1, 1)
        self.pbDirectoryOpen_3 = QtWidgets.QPushButton(self.layoutWidget)
        self.pbDirectoryOpen_3.setObjectName("pbDirectoryOpen_3")
        self.gridLayout_11.addWidget(self.pbDirectoryOpen_3, 1, 5, 1, 1)
        self.lineEditDirInLeaf = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditDirInLeaf.setObjectName("lineEditDirInLeaf")
        self.gridLayout_11.addWidget(self.lineEditDirInLeaf, 1, 2, 1, 1)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.gridLayout_12 = QtWidgets.QGridLayout()
        self.gridLayout_12.setObjectName("gridLayout_12")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_12.addItem(spacerItem6, 1, 0, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.layoutWidget)
        self.label_30.setMinimumSize(QtCore.QSize(120, 0))
        self.label_30.setMaximumSize(QtCore.QSize(143, 16777215))
        self.label_30.setObjectName("label_30")
        self.gridLayout_12.addWidget(self.label_30, 3, 1, 1, 1)
        self.cbFilledDir = QtWidgets.QCheckBox(self.layoutWidget)
        self.cbFilledDir.setChecked(False)
        self.cbFilledDir.setObjectName("cbFilledDir")
        self.gridLayout_12.addWidget(self.cbFilledDir, 1, 1, 1, 1)
        self.sbConvexDir = QtWidgets.QSpinBox(self.layoutWidget)
        self.sbConvexDir.setMinimumSize(QtCore.QSize(85, 0))
        self.sbConvexDir.setMaximumSize(QtCore.QSize(85, 16777215))
        self.sbConvexDir.setPrefix("")
        self.sbConvexDir.setMinimum(0)
        self.sbConvexDir.setMaximum(20)
        self.sbConvexDir.setSingleStep(5)
        self.sbConvexDir.setProperty("value", 0)
        self.sbConvexDir.setDisplayIntegerBase(10)
        self.sbConvexDir.setObjectName("sbConvexDir")
        self.gridLayout_12.addWidget(self.sbConvexDir, 3, 2, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(300, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_12.addItem(spacerItem7, 0, 3, 1, 1)
        self.cbContourDir = QtWidgets.QCheckBox(self.layoutWidget)
        self.cbContourDir.setObjectName("cbContourDir")
        self.gridLayout_12.addWidget(self.cbContourDir, 2, 1, 1, 1)
        self.cbValueDir = QtWidgets.QCheckBox(self.layoutWidget)
        self.cbValueDir.setChecked(False)
        self.cbValueDir.setObjectName("cbValueDir")
        self.gridLayout_12.addWidget(self.cbValueDir, 1, 3, 1, 1)
        self.cbMaskDir = QtWidgets.QCheckBox(self.layoutWidget)
        self.cbMaskDir.setChecked(False)
        self.cbMaskDir.setObjectName("cbMaskDir")
        self.gridLayout_12.addWidget(self.cbMaskDir, 0, 1, 1, 1)
        self.horizontalLayout_20.addLayout(self.gridLayout_12)
        self.gridLayout_11.addLayout(self.horizontalLayout_20, 2, 2, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_11.addItem(spacerItem8, 10, 0, 1, 1)
        self.gridLayout_13 = QtWidgets.QGridLayout()
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.gridLayout_11.addLayout(self.gridLayout_13, 5, 2, 1, 1)
        self.pbRun_3 = QtWidgets.QPushButton(self.layoutWidget)
        self.pbRun_3.setObjectName("pbRun_3")
        self.gridLayout_11.addWidget(self.pbRun_3, 11, 6, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_directory = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab_directory.sizePolicy().hasHeightForWidth())
        self.tab_directory.setSizePolicy(sizePolicy)
        self.tab_directory.setObjectName("tab_directory")
        self.formLayout = QtWidgets.QFormLayout(self.tab_directory)
        self.formLayout.setObjectName("formLayout")
        spacerItem9 = QtWidgets.QSpacerItem(20, 95, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout.setItem(1, QtWidgets.QFormLayout.LabelRole, spacerItem9)
        spacerItem10 = QtWidgets.QSpacerItem(90, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.formLayout.setItem(4, QtWidgets.QFormLayout.LabelRole, spacerItem10)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.lineEditDirOut = QtWidgets.QLineEdit(self.tab_directory)
        self.lineEditDirOut.setObjectName("lineEditDirOut")
        self.gridLayout.addWidget(self.lineEditDirOut, 10, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.tab_directory)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setMaximumSize(QtCore.QSize(28, 16777215))
        self.label_6.setText("")
        self.label_6.setScaledContents(False)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 11, 2, 1, 1)
        self.pbDirectoryExport = QtWidgets.QPushButton(self.tab_directory)
        self.pbDirectoryExport.setObjectName("pbDirectoryExport")
        self.gridLayout.addWidget(self.pbDirectoryExport, 10, 5, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(77, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem11, 2, 6, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_4 = QtWidgets.QLabel(self.tab_directory)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAutoFillBackground(True)
        self.label_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_7.addWidget(self.label_4)
        self.gridLayout.addLayout(self.horizontalLayout_7, 0, 2, 1, 1)
        spacerItem12 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem12, 1, 0, 1, 1)
        spacerItem13 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem13, 0, 0, 1, 1)
        spacerItem14 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem14, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tab_directory)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setAutoFillBackground(True)
        self.label_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 2, 1, 1)
        spacerItem15 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem15, 3, 0, 1, 1)
        self.pbDirectoryOpen = QtWidgets.QPushButton(self.tab_directory)
        self.pbDirectoryOpen.setObjectName("pbDirectoryOpen")
        self.gridLayout.addWidget(self.pbDirectoryOpen, 1, 5, 1, 1)
        self.lineEditDirIn = QtWidgets.QLineEdit(self.tab_directory)
        self.lineEditDirIn.setObjectName("lineEditDirIn")
        self.gridLayout.addWidget(self.lineEditDirIn, 1, 2, 1, 1)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        spacerItem16 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem16, 1, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.tab_directory)
        self.label_8.setMinimumSize(QtCore.QSize(120, 0))
        self.label_8.setMaximumSize(QtCore.QSize(143, 16777215))
        self.label_8.setObjectName("label_8")
        self.gridLayout_5.addWidget(self.label_8, 3, 1, 1, 1)
        self.cbYellowDir = QtWidgets.QCheckBox(self.tab_directory)
        self.cbYellowDir.setChecked(False)
        self.cbYellowDir.setObjectName("cbYellowDir")
        self.gridLayout_5.addWidget(self.cbYellowDir, 1, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.tab_directory)
        self.label_7.setMaximumSize(QtCore.QSize(130, 16777215))
        self.label_7.setObjectName("label_7")
        self.gridLayout_5.addWidget(self.label_7, 0, 1, 1, 1)
        self.sbCropDir = QtWidgets.QSpinBox(self.tab_directory)
        self.sbCropDir.setMaximumSize(QtCore.QSize(85, 16777215))
        self.sbCropDir.setMaximum(100)
        self.sbCropDir.setSingleStep(10)
        self.sbCropDir.setProperty("value", 20)
        self.sbCropDir.setObjectName("sbCropDir")
        self.gridLayout_5.addWidget(self.sbCropDir, 0, 2, 1, 1)
        self.sbBlobDir = QtWidgets.QSpinBox(self.tab_directory)
        self.sbBlobDir.setMinimumSize(QtCore.QSize(85, 0))
        self.sbBlobDir.setMaximumSize(QtCore.QSize(85, 16777215))
        self.sbBlobDir.setPrefix("")
        self.sbBlobDir.setMaximum(200)
        self.sbBlobDir.setSingleStep(10)
        self.sbBlobDir.setProperty("value", 50)
        self.sbBlobDir.setObjectName("sbBlobDir")
        self.gridLayout_5.addWidget(self.sbBlobDir, 3, 2, 1, 1)
        spacerItem17 = QtWidgets.QSpacerItem(300, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem17, 0, 3, 1, 1)
        self.cbCirclesDir = QtWidgets.QCheckBox(self.tab_directory)
        self.cbCirclesDir.setObjectName("cbCirclesDir")
        self.gridLayout_5.addWidget(self.cbCirclesDir, 2, 1, 1, 1)
        self.horizontalLayout_8.addLayout(self.gridLayout_5)
        self.gridLayout.addLayout(self.horizontalLayout_8, 2, 2, 1, 1)
        spacerItem18 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem18, 10, 0, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout.addLayout(self.gridLayout_4, 5, 2, 1, 1)
        self.formLayout.setLayout(4, QtWidgets.QFormLayout.FieldRole, self.gridLayout)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pbRun = QtWidgets.QPushButton(self.tab_directory)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.pbRun.sizePolicy().hasHeightForWidth())
        self.pbRun.setSizePolicy(sizePolicy)
        self.pbRun.setMaximumSize(QtCore.QSize(16777206, 16777215))
        self.pbRun.setObjectName("pbRun")
        self.gridLayout_3.addWidget(self.pbRun, 1, 1, 1, 1)
        spacerItem19 = QtWidgets.QSpacerItem(447, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem19, 1, 2, 1, 1)
        spacerItem20 = QtWidgets.QSpacerItem(409, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem20, 1, 0, 1, 1)
        spacerItem21 = QtWidgets.QSpacerItem(20, 23, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem21, 0, 0, 1, 1)
        self.formLayout.setLayout(5, QtWidgets.QFormLayout.FieldRole, self.gridLayout_3)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.formLayout.setLayout(16, QtWidgets.QFormLayout.LabelRole, self.horizontalLayout_14)
        self.tabWidget.addTab(self.tab_directory, "")
        self.verticalLayout_2.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1133, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.menuFile.setFont(font)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionDirectory = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.actionDirectory.setFont(font)
        self.actionDirectory.setObjectName("actionDirectory")
        self.actionClose = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.actionClose.setFont(font)
        self.actionClose.setObjectName("actionClose")
        self.actionSingle = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.actionSingle.setFont(font)
        self.actionSingle.setObjectName("actionSingle")
        self.menuFile.addAction(self.actionSingle)
        self.menuFile.addAction(self.actionDirectory)
        self.menuFile.addAction(self.actionClose)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Flower Segmentation Tool"))
        self.label_3.setText(_translate("MainWindow", "Crop Image"))
        self.label.setText(_translate("MainWindow", "Image Size"))
        self.label_13.setText(_translate("MainWindow", " in %"))
        self.label_11.setText(_translate("MainWindow", "Flower Detection"))
        self.cbYellow.setText(_translate("MainWindow", "yellow segmentation"))
        self.cbCircle.setText(_translate("MainWindow", "circle around flowers"))
        self.label_12.setText(_translate("MainWindow", "in px"))
        self.label_2.setText(_translate("MainWindow", "Min Flower Size"))
        self.label_15.setText(_translate("MainWindow", "Leaf Herbivory"))
        self.cbGreen.setText(_translate("MainWindow", "green segmentation"))
        self.cbMask.setText(_translate("MainWindow", "leaf mask"))
        self.cbContours.setText(_translate("MainWindow", "leaf contours"))
        self.pbExport.setText(_translate("MainWindow", "Export"))
        self.pbUpdate.setText(_translate("MainWindow", "Update"))
        self.lineEditImage.setPlaceholderText(_translate("MainWindow", "/home/../image.jpg "))
        self.pbImageOpen.setText(_translate("MainWindow", "Open"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_example), _translate("MainWindow", "Plot"))
        self.lineEditDirOutLeaf.setPlaceholderText(_translate("MainWindow", "/home/OutputDirectory"))
        self.pbDirectoryExport_3.setText(_translate("MainWindow", "Export"))
        self.label_28.setText(_translate("MainWindow", "Import all Images from Directory:"))
        self.label_29.setText(_translate("MainWindow", "Export to Directory:"))
        self.pbDirectoryOpen_3.setText(_translate("MainWindow", "Import"))
        self.lineEditDirInLeaf.setPlaceholderText(_translate("MainWindow", "/home/../InputDirectory"))
        self.label_30.setText(_translate("MainWindow", "convex hull diff"))
        self.cbFilledDir.setText(_translate("MainWindow", "filled leaf image"))
        self.cbContourDir.setText(_translate("MainWindow", "contour and convex hull"))
        self.cbValueDir.setText(_translate("MainWindow", "value export"))
        self.cbMaskDir.setText(_translate("MainWindow", "mask image"))
        self.pbRun_3.setText(_translate("MainWindow", "Run"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Herbivory"))
        self.lineEditDirOut.setPlaceholderText(_translate("MainWindow", "/home/OutputDirectory"))
        self.pbDirectoryExport.setText(_translate("MainWindow", "Export"))
        self.label_4.setText(_translate("MainWindow", "Import all Images from Directory:"))
        self.label_5.setText(_translate("MainWindow", "Export to Directory:"))
        self.pbDirectoryOpen.setText(_translate("MainWindow", "Import"))
        self.lineEditDirIn.setPlaceholderText(_translate("MainWindow", "/home/../InputDirectory"))
        self.label_8.setText(_translate("MainWindow", "Min Flower Size in px"))
        self.cbYellowDir.setText(_translate("MainWindow", "yellow segmentation"))
        self.label_7.setText(_translate("MainWindow", "Crop image to %"))
        self.cbCirclesDir.setText(_translate("MainWindow", "circle around flowers"))
        self.pbRun.setText(_translate("MainWindow", "Run"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_directory), _translate("MainWindow", "Flowers"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionDirectory.setText(_translate("MainWindow", "Open Folder"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionSingle.setText(_translate("MainWindow", "Open Image"))
from gui.pyqt5backend.mplwidget_nav import MplWidget