import cv2
import sys
import torch
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget, QHeaderView
from PySide6.QtCore import QTimer, Qt, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem, QResizeEvent
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
    signal_update_video_stream = Signal(QPixmap)
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # Call setupUi to initialize ui variable
        self.init_table()
        self.bind_slot()
        self.cap = cv2.VideoCapture(0)
        self.cap.read()
        self.run_flag = False
        # 先准备一个线程空间，防止用户一次都没有打开摄像头就点击关闭程序时，closeEvent函数报错
        self.detect_thread = None
        # 绑定信号和槽函数
        self.signal_update_current_table.connect(self.update_current_table)
        self.signal_update_video_stream.connect(self.update_video_stream)
        # 创建一个池子result_pool并设置池子的pool_size大小(存储多少组数据),池子里任何一个物体左上角的坐标和右下角的坐标差都在+-allow_range内，那么就认为识别成功
        self.allow_range = 30
        self.pool_size = 20
        self.data_pool = []

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
        headers = ['类别', '物体数量', '满载检测']
        self.model_history.setHorizontalHeaderLabels(headers)
        # 添加示例数据
        data = [
            ['可回收垃圾', 2, '未满载'],
            ['不可回收垃圾', 5, '不检测'],
            ['厨余垃圾', 3, '不检测'],
            ['有害垃圾', 4, '不检测']
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
            self.run_flag = False
        else:
            self.pushButton.setText("停止识别")
            self.label_statement.setText('状态：识别中')
            self.run_flag = True
            # 开一个线程运行detect函数，否则ui会卡死
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
                # t1和t2用于计算fps
                t1 = time.time()
                # 模型推理
                det_boxes, scores, ids = infer_img(
                    img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4, thred_cond=0.4)
                t2 = time.time()
                for box, score, id in zip(det_boxes, scores, ids):
                    label = '%s:%.2f' % (dic_labels[id], score)
                    plot_one_box(box.astype(np.int16), img0, color=(
                        255, 0, 0), label=label, line_thickness=None)
                str_FPS = "FPS: %.2f" % (1./(t2-t1))
                # 在图像上添加fps信息，字体为等线
                cv2.putText(img0, str_FPS, (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

                # 把图像显示到label_videoStream上
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # 将帧转换为RGB格式
                height, width, channel = img0.shape
                bytesPerLine = 3 * width
                qImg = QImage(img0.data, width, height,
                              bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                # 图像左右翻转
                # pixmap = pixmap.transformed(QtGui.QTransform().scale(-1, 1))
                # 发送信号，更新实时视频流
                self.signal_update_video_stream.emit(pixmap)

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

                # 静止检测
                self.static_detect(data)

                cv2.waitKey(1)
            else:
                QMessageBox.warning(self, '警告', '摄像头打开失败，请检查摄像头是否连接正常！')
                self.run_flag = False
                break
        self.pushButton.setEnabled(True)
        self.pushButton.setText("开始识别")

    def static_detect(self, data):
        if data == []:
            # 如果data为空，清空池子，然后返回
            self.data_pool.clear()
            return
        elif len(self.data_pool) == 0:
            # 如果池子为空，那么直接添加数据
            self.data_pool.append(data)
            return
        elif len(data) != len(self.data_pool[len(self.data_pool)-1]):
            # 如果池子最后一组数据的长度（物体数量）和当前数据长度（物体数量）不一致，就直接清空池子，然后添加数据
            self.data_pool.clear()
            self.data_pool.append(data)
            return
        else:
            # 如果池子最后一组数据的长度（物体数量）和当前数据长度（物体数量）一致，就尝试进行排序/对应
            sorted_flag, processed_data = self.sort_object(data)
            if not sorted_flag:
                # 如果排序/对应失败，就直接清空池子，然后添加本次识别的数据
                self.data_pool.clear()
                self.data_pool.append(data)
                # 或者是self.data_pool.append(processed_data)也一样
                return
            else:
                # 排序/对应成功
                self.data_pool.append(processed_data)
                # 如果池子没满，就直接返回
                if len(self.data_pool) < self.pool_size:
                    return
                else:
                    # 如果池子满了，就检查池子中各组数据的每一个物体的左上角的坐标和右下角的坐标差的最大值是否都在+-allow_range内
                    # 如果都在+-allow_range内，就认为识别成功（该池子是“成功的池子”），否则就认为识别失败
                    if not self.successful_pool():
                        # 删除最早一次识别的数据
                        self.data_pool.pop(0)
                        return
                    else:
                        sucessful_data = self.data_pool[-1]
                        self.label_statement.setText('状态：识别成功')
                        # 清空池子，为下一次识别做准备
                        self.data_pool.clear()
                        # 执行通信逻辑并进入消息循环，暂时用打印代替
                        print(sucessful_data)
                        # 停止识别（暂时先写成停止
                        if self.pushButton.isEnabled():
                            self.pushButton.click()
                        return


    def sort_object(self, data):
        sorted_data = []
        pool_last_data = self.data_pool[-1]
        # 给sorted_data创建一个和pool_last_data一样的空列表，用于存储排序/对应后的数据
        for i in range(len(pool_last_data)):
            sorted_data.append([])
        # 把data中的每一个物体和池子中最后一组数据的每一个物体进行比对，二者的左上角的坐标和右下角的坐标差值都在+-allow_range内
        # 那么就认为是同一个物体,并把data中的物体添加到sorted_data对应的位置（索引编号）上
        for row_data in data:
            for pool_row_data in pool_last_data:
                if (abs(row_data[2][0]-pool_row_data[2][0]) < self.allow_range and 
                    abs(row_data[2][1]-pool_row_data[2][1]) < self.allow_range and 
                    abs(row_data[2][0]+row_data[3][0]-pool_row_data[2][0]-pool_row_data[3][0]) < self.allow_range and 
                    abs(row_data[2][1]+row_data[3][1]-pool_row_data[2][1]-pool_row_data[3][1]) < self.allow_range):
                    # 直接把sorted_data中对应位置的空列表换成row_data
                    sorted_data[pool_last_data.index(pool_row_data)] = row_data
                    break            
            else:
                # 如果把池子里的物体遍历完了还没有找到对应的物体，直接返回失败和原列表data
                # Python中的for循环可以带else语句，当for循环正常结束时，else语句会执行，当for循环被break时，else语句不会执行
                return False, data
        else:
            # 如果data中的每一个物体都找到了对应的物体，那么就返回成功和sorted_data
            return True, sorted_data

    def successful_pool(self):
        # 检查池子中各组数据的每一个物体的左上角的坐标和右下角的坐标差的最大值是否都在+-allow_range内
        # 如果都在+-allow_range内，就认为识别成功（该池子是“成功的池子”）返回true，否则就认为识别失败
        for i in range(len(self.data_pool)-1):
            for j in range(len(self.data_pool[i])):
                if (abs(self.data_pool[i][j][2][0]-self.data_pool[i+1][j][2][0]) > self.allow_range or 
                    abs(self.data_pool[i][j][2][1]-self.data_pool[i+1][j][2][1]) > self.allow_range or 
                    abs(self.data_pool[i][j][2][0]+self.data_pool[i][j][3][0]-self.data_pool[i+1][j][2][0]-self.data_pool[i+1][j][3][0]) > self.allow_range or
                    abs(self.data_pool[i][j][2][1]+self.data_pool[i][j][3][1]-self.data_pool[i+1][j][2][1]-self.data_pool[i+1][j][3][1]) > self.allow_range):
                    return False
        else:
            return True

    def update_current_table(self, data):
        # 把模型里的数据清空
        self.model_current.removeRows(0, self.model_current.rowCount())
        # 更新实时表格模型
        for rowData in data:
            self.model_current.appendRow(
                [QStandardItem(str(value)) for value in rowData])
            for col in range(6):
                self.model_current.item(self.model_current.rowCount(
                ) - 1, col).setTextAlignment(Qt.AlignCenter)

    def update_video_stream(self, pixmap):
        self.label_videoStream.setPixmap(pixmap)

    def resizeEvent(self, event: QResizeEvent) -> None:
        # 计算widget比例，使得画面可以自适应的同时保持锁定4:3比例
        ratio = 4 / 3
        w = self.widget.width()
        h = self.widget.height()
        if w / h > ratio:
            # 如果宽高比大于4:3，宽度太大，调整宽度
            self.label_videoStream.resize(h * ratio, h)
            # 居中显示
            self.label_videoStream.move(w / 2 - h * ratio / 2, 0)
        else:
            # 如果宽高比小于4:3，高度太大，调整高度
            self.label_videoStream.resize(w, w / ratio)
            # 居中显示
            self.label_videoStream.move(0, h / 2 - w / ratio / 2)
        return super().resizeEvent(event)

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
    window.showMaximized()
    app.exec()
