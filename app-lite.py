import cv2
import sys
import torch
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget, QHeaderView
from PySide6.QtCore import QTimer, Qt, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem
from AppGui_ui import Ui_Form
import numpy as np
import onnxruntime as ort
import time
import random
import threading


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def _make_grid(nx, ny):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)


def cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride):
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w / stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)
        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs


def post_process_opencv(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
    conf = outputs[:, 4].tolist()
    c_x = outputs[:, 0]/model_w*img_w
    c_y = outputs[:, 1]/model_h*img_h
    w = outputs[:, 2]/model_w*img_w
    h = outputs[:, 3]/model_h*img_h
    p_cls = outputs[:, 5:]
    if len(p_cls.shape) == 1:
        p_cls = np.expand_dims(p_cls, 1)
    cls_id = np.argmax(p_cls, axis=1)
    p_x1 = np.expand_dims(c_x-w/2, -1)
    p_y1 = np.expand_dims(c_y-h/2, -1)
    p_x2 = np.expand_dims(c_x+w/2, -1)
    p_y2 = np.expand_dims(c_y+h/2, -1)
    areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)
    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
    if len(ids) > 0:
        return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
    else:
        return [], [], []


def infer_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4, thred_cond=0.2):
    # 图像预处理
    img = cv2.resize(img0, [model_w, model_h], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
    # 输出坐标矫正
    outs = cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride)
    # 检测框计算
    img_h, img_w, _ = np.shape(img0)
    boxes, confs, ids = post_process_opencv(
        outs, model_h, model_w, img_h, img_w, thred_nms, thred_cond)
    return boxes, confs, ids


class MainWindow(QWidget, Ui_Form):
    # 定义一个信号，用于更新实时表格
    # 不能在子线程中直接调用ui的控件（否则会出bug），所以需要定义一个信号，用于在子线程中发送信号，然后在主线程中接收信号，从而更新ui
    signal_update_current_table = Signal(list)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # Call setupUi to initialize ui variable
        self.init_table()
        self.bind_slot()
        self.cap = cv2.VideoCapture(0)
        self.run_flag = False
        # 先准备一个线程空间，防止用户一次都没有打开摄像头就点击关闭程序时，closeEvent函数报错
        self.detect_thread = None
        # 绑定信号和槽函数
        self.signal_update_current_table.connect(self.update_current_table)

    def init_table(self):
        # 创建实时表格模型
        self.model_current = QStandardItemModel()
        self.tableView_current.setModel(self.model_current)
        # 设置表头
        headers = ['物体序号', '物体类别', '物体坐标', '物体尺寸', '置信度', '分类状态']
        self.model_current.setHorizontalHeaderLabels(headers)
        # 添加示例数据
        data = [
            [1, '可回收', (100, 200), (100, 100), 0.9, '完成'],
            [2, '不可回收', (100, 200), (100, 100), 0.8, '分类中'],
            [3, '厨余', (100, 200), (100, 100), 0.7, '等待'],
            [4, '有害', (100, 200), (100, 100), 0.6, '等待']
        ]
        for row, rowData in enumerate(data):
            for col, value in enumerate(rowData):
                item = QStandardItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.model_current.setItem(row, col, item)
        # 调整列宽
        self.tableView_current.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # #表格文字格式
        # self.tableView.setStyleSheet("QTableView{font-family:'Microsoft YaHei';font-size:12px;color:rgb(255,0,0);}")
        # 设置表头和示例数据所有文字水平居中
        self.tableView_current.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)

        # 创建历史表格模型
        self.model_history = QStandardItemModel()
        self.tableView_history.setModel(self.model_history)
        # 设置表头
        headers = ['类别', '物体数量']
        self.model_history.setHorizontalHeaderLabels(headers)
        # 添加示例数据
        data = [
            ['可回收垃圾', 2],
            ['不可回收垃圾', 5],
            ['厨余垃圾', 3],
            ['有害垃圾', 4]
        ]
        for row, rowData in enumerate(data):
            for col, value in enumerate(rowData):
                item = QStandardItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.model_history.setItem(row, col, item)
        # 调整列宽
        self.tableView_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # #表格文字格式
        # self.tableView.setStyleSheet("QTableView{font-family:'Microsoft YaHei';font-size:12px;color:rgb(255,0,0);}")
        # 设置表头和示例数据所有文字水平居中
        self.tableView_history.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)

    def open_video(self):
        if self.run_flag == True:
            self.pushButton.setText("请稍候...")
            self.pushButton.setEnabled(False)
            # self.pushButton.setText("打开摄像头")
            self.run_flag = False
        else:
            self.pushButton.setText("停止识别")
            self.run_flag = True
            # 开一个线程运行detect函数
            self.detect_thread = threading.Thread(target=self.detect)
            self.detect_thread.start()

    def detect(self):
        while self.run_flag:
            success, img0 = self.cap.read()
            if success:
                # 裁剪视频帧到4:3比例
                height, width, _ = img0.shape
                if width / height > 4 / 3:
                    new_width = int(height * 4 / 3)
                    left = int((width - new_width) / 2)
                    right = left + new_width
                    img0 = img0[:, left:right]
                else:
                    new_height = int(width * 3 / 4)
                    top = int((height - new_height) / 2)
                    bottom = top + new_height
                    img0 = img0[top:bottom, :]

                t1 = time.time()
                det_boxes, scores, ids = infer_img(
                    img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4, thred_cond=0.4)
                t2 = time.time()
                for box, score, id in zip(det_boxes, scores, ids):
                    label = '%s:%.2f' % (dic_labels[id], score)
                    plot_one_box(box.astype(np.int16), img0, color=(
                        255, 0, 0), label=label, line_thickness=None)
                str_FPS = "FPS: %.2f" % (1./(t2-t1))
                # 在图像上添加fps信息，字体为等线
                # cv2.putText(img0, str_FPS, (20, 40),
                #             cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                self.label_statement.setText('状态：识别中。' + str_FPS)

                # 把图像显示到label_videoStream上
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # 将帧转换为RGB格式
                height, width, channel = img0.shape
                bytesPerLine = 3 * width
                qImg = QImage(img0.data, width, height,
                              bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                # 图像左右翻转
                # pixmap = pixmap.transformed(QtGui.QTransform().scale(-1, 1))
                self.label_videoStream.setPixmap(pixmap)

                # 把结果添加data，按照['物体序号', '物体类别', '坐标', '尺寸','置信度', '分类状态']六列顺序添加
                rank = 1
                data = []
                for box, score, id in zip(det_boxes, scores, ids):
                    row_data = [rank, dic_labels[id], (int(box[0]), int(box[1])), (int(
                        box[2]-box[0]), int(box[3]-box[1])), f'{score:.2f}', '等待']
                    data.append(row_data)
                    rank += 1
                # 发送信号，更新实时表格
                self.signal_update_current_table.emit(data)
                cv2.waitKey(1)
            else:
                QMessageBox.warning(self, '警告', '摄像头打开失败，请检查摄像头是否连接正常！')
                self.run_flag = False
                break
        self.pushButton.setEnabled(True)
        self.pushButton.setText("开始识别")

    def update_current_table(self, data):
        self.model_current.clear()
        headers = ['物体序号', '物体类别', '物体坐标', '物体尺寸', '置信度', '分类状态']
        self.model_current.setHorizontalHeaderLabels(headers)
        # 更新实时表格模型
        for rowData in data:
            self.model_current.appendRow(
                [QStandardItem(str(value)) for value in rowData])
            for col in range(6):
                self.model_current.item(self.model_current.rowCount(
                ) - 1, col).setTextAlignment(Qt.AlignCenter)
        # 如果模型为空那就添加一行空数据
        if self.model_current.rowCount() == 0:
            self.model_current.appendRow(QStandardItem(''))

    def closeEvent(self, event):
        # 停止detect函数
        self.run_flag = False
        self.cap.release()
        # 退出程序
        super().closeEvent(event)

    def bind_slot(self):
        self.pushButton.clicked.connect(self.open_video)


if __name__ == "__main__":
    # 模型加载
    model_pb_path = "best.onnx"
    so = ort.SessionOptions()
    net = ort.InferenceSession(model_pb_path, so)
    # 标签字典
    dic_labels = {0: 'WaterBottle'}
    # 模型参数
    model_h = 320
    model_w = 320
    nl = 3
    na = 3
    stride = [8., 16., 32.]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62,
                                          45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
