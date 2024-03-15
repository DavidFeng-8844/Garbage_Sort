# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'AppGui.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QGridLayout,
    QHeaderView, QLabel, QPushButton, QSizePolicy,
    QSpacerItem, QTableView, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.setWindowModality(Qt.NonModal)
        Form.resize(758, 496)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(9, -1, 9, 9)
        self.widget_video = QWidget(Form)
        self.widget_video.setObjectName(u"widget_video")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_video.sizePolicy().hasHeightForWidth())
        self.widget_video.setSizePolicy(sizePolicy)
        self.widget_video.setAutoFillBackground(False)
        self.label_videoStream = QLabel(self.widget_video)
        self.label_videoStream.setObjectName(u"label_videoStream")
        self.label_videoStream.setGeometry(QRect(0, 0, 225, 45))
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_videoStream.sizePolicy().hasHeightForWidth())
        self.label_videoStream.setSizePolicy(sizePolicy1)
        font = QFont()
        font.setPointSize(26)
        self.label_videoStream.setFont(font)
        self.label_videoStream.setScaledContents(True)
        self.label_videoStream.setAlignment(Qt.AlignCenter)
        self.label_countdown = QLabel(self.widget_video)
        self.label_countdown.setObjectName(u"label_countdown")
        self.label_countdown.setGeometry(QRect(0, 40, 191, 20))
        sizePolicy1.setHeightForWidth(self.label_countdown.sizePolicy().hasHeightForWidth())
        self.label_countdown.setSizePolicy(sizePolicy1)
        font1 = QFont()
        font1.setPointSize(12)
        self.label_countdown.setFont(font1)
        self.label_countdown.setAutoFillBackground(False)
        self.label_countdown.setStyleSheet(u"QLabel{background-color:rgba(240, 240, 240,0.8);}")
        self.label_countdown.setIndent(4)

        self.gridLayout.addWidget(self.widget_video, 3, 1, 4, 1)

        self.label_current = QLabel(Form)
        self.label_current.setObjectName(u"label_current")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_current.sizePolicy().hasHeightForWidth())
        self.label_current.setSizePolicy(sizePolicy2)
        font2 = QFont()
        font2.setPointSize(16)
        self.label_current.setFont(font2)

        self.gridLayout.addWidget(self.label_current, 3, 2, 1, 1)

        self.label_statement = QLabel(Form)
        self.label_statement.setObjectName(u"label_statement")
        self.label_statement.setFont(font2)
        self.label_statement.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_statement, 3, 3, 1, 1)

        self.label_fullDetect = QLabel(Form)
        self.label_fullDetect.setObjectName(u"label_fullDetect")
        self.label_fullDetect.setFont(font2)

        self.gridLayout.addWidget(self.label_fullDetect, 5, 3, 1, 2)

        self.tableView_current = QTableView(Form)
        self.tableView_current.setObjectName(u"tableView_current")
        sizePolicy3 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.tableView_current.sizePolicy().hasHeightForWidth())
        self.tableView_current.setSizePolicy(sizePolicy3)
        self.tableView_current.setMinimumSize(QSize(600, 0))
        self.tableView_current.setFont(font1)
        self.tableView_current.setStyleSheet(u"")
        self.tableView_current.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tableView_current.setAutoScroll(True)
        self.tableView_current.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tableView_current.verticalHeader().setVisible(False)

        self.gridLayout.addWidget(self.tableView_current, 4, 2, 1, 3)

        self.pushButton = QPushButton(Form)
        self.pushButton.setObjectName(u"pushButton")
        sizePolicy4 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy4)
        self.pushButton.setMinimumSize(QSize(100, 0))
        font3 = QFont()
        font3.setPointSize(11)
        self.pushButton.setFont(font3)

        self.gridLayout.addWidget(self.pushButton, 3, 4, 1, 1)

        self.label_history = QLabel(Form)
        self.label_history.setObjectName(u"label_history")
        sizePolicy2.setHeightForWidth(self.label_history.sizePolicy().hasHeightForWidth())
        self.label_history.setSizePolicy(sizePolicy2)
        self.label_history.setFont(font2)

        self.gridLayout.addWidget(self.label_history, 5, 2, 1, 1)

        self.tableView_history = QTableView(Form)
        self.tableView_history.setObjectName(u"tableView_history")
        sizePolicy2.setHeightForWidth(self.tableView_history.sizePolicy().hasHeightForWidth())
        self.tableView_history.setSizePolicy(sizePolicy2)
        self.tableView_history.setMinimumSize(QSize(0, 0))
        self.tableView_history.setMaximumSize(QSize(16777215, 151))
        self.tableView_history.setFont(font1)
        self.tableView_history.setStyleSheet(u"")
        self.tableView_history.setFrameShape(QFrame.StyledPanel)

        self.gridLayout.addWidget(self.tableView_history, 6, 2, 1, 1)

        self.widget_fullDetect = QWidget(Form)
        self.widget_fullDetect.setObjectName(u"widget_fullDetect")
        self.widget_fullDetect.setStyleSheet(u"QWidget{background-color:rgb(255, 255, 255);}\n"
"QWidget{border:1px solid gray;}")
        self.gridLayout_2 = QGridLayout(self.widget_fullDetect)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 1, 0, 1, 1)

        self.label_fullState = QLabel(self.widget_fullDetect)
        self.label_fullState.setObjectName(u"label_fullState")
        sizePolicy4.setHeightForWidth(self.label_fullState.sizePolicy().hasHeightForWidth())
        self.label_fullState.setSizePolicy(sizePolicy4)
        self.label_fullState.setMinimumSize(QSize(80, 80))
        self.label_fullState.setMaximumSize(QSize(80, 80))
        font4 = QFont()
        font4.setPointSize(16)
        font4.setBold(True)
        self.label_fullState.setFont(font4)
        self.label_fullState.setStyleSheet(u"QLabel{border-radius:40px;}\n"
"QLabel{border:5px solid green;}\n"
"QLabel{color:green;}")
        self.label_fullState.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.label_fullState, 1, 1, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_2, 1, 2, 1, 1)

        self.label_fullInfo = QLabel(self.widget_fullDetect)
        self.label_fullInfo.setObjectName(u"label_fullInfo")
        sizePolicy1.setHeightForWidth(self.label_fullInfo.sizePolicy().hasHeightForWidth())
        self.label_fullInfo.setSizePolicy(sizePolicy1)
        self.label_fullInfo.setFont(font3)
        self.label_fullInfo.setStyleSheet(u"QLabel{border-radius:10px;}")
        self.label_fullInfo.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.label_fullInfo, 2, 0, 1, 3)


        self.gridLayout.addWidget(self.widget_fullDetect, 6, 3, 1, 2)

        self.label_title = QLabel(Form)
        self.label_title.setObjectName(u"label_title")
        sizePolicy2.setHeightForWidth(self.label_title.sizePolicy().hasHeightForWidth())
        self.label_title.setSizePolicy(sizePolicy2)
        font5 = QFont()
        font5.setPointSize(20)
        font5.setBold(True)
        self.label_title.setFont(font5)
        self.label_title.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_title, 0, 1, 1, 4)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"\u5b9e\u65f6\u5783\u573e\u5206\u7c7b\u7cfb\u7edf", None))
        self.label_videoStream.setText(QCoreApplication.translate("Form", u"Video Stream", None))
        self.label_countdown.setText(QCoreApplication.translate("Form", u"\u5ba3\u4f20\u7247\u8d85\u65f6\u64ad\u653e\u5269\u4f59\uff1a", None))
        self.label_current.setText(QCoreApplication.translate("Form", u" \u5b9e\u65f6\u4fe1\u606f\uff1a", None))
        self.label_statement.setText(QCoreApplication.translate("Form", u"\u72b6\u6001\uff1a\u505c\u6b62", None))
        self.label_fullDetect.setText(QCoreApplication.translate("Form", u"\u6ee1\u8f7d\u68c0\u6d4b\uff1a", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"\u5f00\u59cb\u8bc6\u522b", None))
        self.label_history.setText(QCoreApplication.translate("Form", u" \u7edf\u8ba1\u4fe1\u606f\uff1a", None))
        self.label_fullState.setText(QCoreApplication.translate("Form", u"\u672a\u6ee1\u8f7d", None))
        self.label_fullInfo.setText(QCoreApplication.translate("Form", u"\u53ef\u56de\u6536\u5783\u573e\u6876\n"
"\u5c1a\u672a\u8fdb\u884c\u6ee1\u8f7d\u68c0\u6d4b...", None))
        self.label_title.setText(QCoreApplication.translate("Form", u"\u5206\u7684\u90fd\u5bf9\u961f-\u5b9e\u65f6\u5783\u573e\u5206\u7c7b\u7cfb\u7edf", None))
    # retranslateUi

