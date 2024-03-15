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
from time import sleep
import datetime
import math

# 运行平台：Rasp/PC
platform = 'PC'
if platform == 'Rasp':
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)

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
        c1 = (c1[0] - 1, c1[1]) # 标签往左移1px，跟左边框对齐
        c2 = (c2[0] - 1, c2[1])
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
    # 定义一个信号，用于接收线程的指令
    # 不能在子线程中直接调用ui的控件或者启停QTimer，否则会出bug/报错
    # 所以需要定义一个信号，用于在子线程中发送信号，然后在主线程中接收信号
    signal_update_current_table = Signal(list)
    signal_update_history_table = Signal(list)
    signal_update_video_stream = Signal(QPixmap)
    signal_control_timer_play_film = Signal(bool)
    signal_control_timer_timeout = Signal(bool)

# 以下函数为初始化函数
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # 初始化ui
        self.init_table()
        self.bind_slot()
        self.cap = cv2.VideoCapture(0)
        self.cap.read() # 读取一下，这样后续调用会比较快
        self.run_flag = False
        # 准备一个线程，防止用户一次都没有打开摄像头就点击关闭程序时，closeEvent函数报错
        self.detect_thread = None
        # 创建一个池子result_pool并设置池子的pool_size大小(存储多少组数据),池子里任何一个物体左上角的坐标和右下角的坐标差都在+-allow_range内，那么就认为识别成功
        self.allow_range = 30
        self.pool_size = 50
        if platform == 'Rasp':
            self.pool_size = 15
        self.data_pool = []

        # 机械臂的当前位置
        self.arm_current_position = [1, 0]
        # 机械臂的归零位置
        self.arm_zero_position = [1, 0]
        # 四种垃圾投放对应的机械臂位置
        self.arm_position_recyclable = [1.1, 1.1]
        self.arm_position_organic = [1.1, -0.1]
        self.arm_position_harmful = [-0.1, -0.1]
        self.arm_position_other = [-0.1, 1.1]
        # 边长对应的步进电机步数
        self.arm_step = 4500

        # 从本地目录加载film.mp4
        self.cap_film = cv2.VideoCapture('film.mp4')
        # 设置一个QTimer用于播放宣传片
        self.timer_play_film = QTimer()
        self.timer_play_film.setInterval(60)
        self.timer_play_film.timeout.connect(self.play_film)
        self.film_on = False
        # 设置一个QTimer用于检测超时（连续n秒没有检测到任何物体就开始播放宣传片）
        self.timer_timeout = QTimer()
        self.timer_timeout.setInterval(100)
        self.timeout_default_count = 8.0
        self.timeout_count = self.timeout_default_count
        self.timer_timeout.timeout.connect(self.no_object_timeout)
        # 开始播放
        self.film_on = True
        self.timer_play_film.start()
        self.label_countdown.setVisible(False)

        # 开始检测
        self.pushButton.click()

    def bind_slot(self):
        # 绑定信号和槽函数
        self.pushButton.clicked.connect(self.open_video)        
        self.signal_update_current_table.connect(self.update_current_table)
        self.signal_update_history_table.connect(self.update_history_table)
        self.signal_update_video_stream.connect(self.update_video_stream)
        self.signal_control_timer_play_film.connect(self.control_timer_play_film)
        self.signal_control_timer_timeout.connect(self.control_timer_timeout)

    def init_table(self):
        # 创建实时表格模型
        self.model_current = QStandardItemModel()
        self.tableView_current.setModel(self.model_current)
        # 设置表头
        headers = ['物体序号', '物体类别', '物体坐标', '物体尺寸', '置信度', '分类状态']
        self.model_current.setHorizontalHeaderLabels(headers)
        # 添加示例数据
        data = [
            [1, '有害', (100, 200), (100, 100), 0.9, '完成'],
            [2, '厨余', (100, 200), (100, 100), 0.8, '分类中'],
            [3, '其他', (100, 200), (100, 100), 0.7, '等待'],
            [4, '可回收', (100, 200), (100, 100), 0.6, '等待']
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
            ['可回收垃圾', 0],
            ['厨余垃圾', 0],
            ['有害垃圾', 0],
            ['其他垃圾', 0]
        ]
        for row, rowData in enumerate(data):
            for col, value in enumerate(rowData):
                item = QStandardItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.model_history.setItem(row, col, item)
        # 调整列宽
        self.tableView_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 表格文字格式
        # self.tableView.setStyleSheet("QTableView{font-family:'Microsoft YaHei';font-size:12px;color:rgb(255,0,0);}")
        # 设置表头和示例数据所有文字水平居中
        self.tableView_history.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
    
    def open_video(self):
        # 相应用户点击开始/停止识别按钮
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

# 以下函数为检测功能实现
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
                    img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4, thred_cond=0.3)
                t2 = time.time()

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
                
                # 如果data不为空，重置 超时计数器
                if data != []:
                    self.timeout_count = self.timeout_default_count
                    # 如果 超时计时 没开始，就开始计时
                    if not self.timer_timeout.isActive:
                        self.signal_control_timer_timeout.emit(True)
                
                # 如果data为空并且宣传片正在播放，就不需要显示实时画面，直接跳过
                # 否则就显示实时画面
                if not (self.film_on and data == []):
                    # 如果data不空，而且宣传片正在播放，就打断播放宣传片
                    if self.film_on:
                        self.signal_control_timer_play_film.emit(False)
                        self.film_on = False
                        self.signal_control_timer_timeout.emit(True)

                    # 在图像上绘制检测框和置信度
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

                # 静止检测
                self.static_detect(data)

                cv2.waitKey(1)
            else:
                QMessageBox.warning(self, '警告', '摄像头打开失败，请检查摄像头是否连接正常！')
                self.run_flag = False
                break

        self.label_statement.setText('状态：停止')
        self.pushButton.setEnabled(True)
        self.pushButton.setText("开始识别")
        # 用户关闭了识别，如果宣传片没在播放，就开始播放宣传片
        if not self.film_on:
            self.signal_control_timer_timeout.emit(False)
            self.timeout_count = self.timeout_default_count
            self.label_countdown.setText('宣传片超时播放剩余：'+str(self.timeout_count))
            self.film_on = True
            self.signal_control_timer_play_film.emit(True)

# 以下函数为静止检测功能实现
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
                        successful_data = self.data_pool[-1]
                        self.label_statement.setText('状态：分类中')
                        # 清空池子，为下一次识别做准备
                        self.data_pool.clear()
                        print(successful_data)
                        
                        # 停止超时计时并设置超时计数器
                        self.signal_control_timer_timeout.emit(False)
                        self.timeout_count = self.timeout_default_count

                        # 控制电机（包含更新历史表格、满载检测）
                        if platform == 'Rasp':
                            self.motor_control(successful_data)
                        
                        # 分类完成
                        self.label_statement.setText('状态：识别中')

                        # 树莓派上需要重新开关一下摄像头，不然会残留一下之前的画面，电脑上不可开启（可能导致摄像头卡住）
                        if platform == 'Rasp':
                            self.cap.release()
                            self.cap = cv2.VideoCapture(0)

                        # 想要分类完成后等待超时再播放的话用这一句
                        self.signal_control_timer_timeout.emit(True)
                        # 想要直接开始播放用以下两句
                        # self.film_on = True
                        # self.signal_control_timer_play_film.emit(True)

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

# 以下函数为电机控制功能实现
    def motor_control(self, successful_data):
        orderd_data = self.classification_order(successful_data)
        current_object = 0
        while current_object < len(orderd_data):
            # 判断剩余物体的类别是否都相同
            same_flag = True
            for i in range(current_object, len(orderd_data)-1):
                if orderd_data[i][1] != orderd_data[i+1][1]:
                    same_flag = False
                    break

            if same_flag:
                # 如果剩余物体的类别都相同，就把机械臂归位，然后直接倾倒
                zero_x = self.arm_zero_position[0]
                zero_y = self.arm_zero_position[1]
                self.arm_go_position(zero_x, zero_y)

                self.platform_control(orderd_data[current_object][1])
                
                # 更新倾倒的物体的分类状态并更新当前表格
                for i in range(current_object, len(orderd_data)):
                    orderd_data[i][5] = '成功'
                self.signal_update_current_table.emit(orderd_data)
                # 更新历史表格
                if orderd_data[current_object][1] == 'Recyclable':
                    item = self.model_history.item(0, 1)
                    item.setText(str(int(item.text())+len(orderd_data)-current_object))
                elif orderd_data[current_object][1] == 'Organic':
                    item = self.model_history.item(1, 1)
                    item.setText(str(int(item.text())+len(orderd_data)-current_object))
                elif orderd_data[current_object][1] == 'Harmful':
                    item = self.model_history.item(2, 1)
                    item.setText(str(int(item.text())+len(orderd_data)-current_object))
                elif orderd_data[current_object][1] == 'Other':
                    item = self.model_history.item(3, 1)
                    item.setText(str(int(item.text())+len(orderd_data)-current_object))

                current_object = len(orderd_data)
                break
            else:
                # 如果剩余物体的类别不同，就抓取当前物体
                self.arm_catch_throw(orderd_data[current_object])
                
                # 更新当前物体的分类状态并更新当前表格
                orderd_data[current_object-1][5] = '成功'
                self.signal_update_current_table.emit(orderd_data)
                # 更新历史表格
                if orderd_data[current_object][1] == 'Recyclable':
                    item = self.model_history.item(0, 1)
                    item.setText(str(int(item.text())+1))
                elif orderd_data[current_object][1] == 'Organic':
                    item = self.model_history.item(1, 1)
                    item.setText(str(int(item.text())+1))
                elif orderd_data[current_object][1] == 'Harmful':
                    item = self.model_history.item(2, 1)
                    item.setText(str(int(item.text())+1))
                elif orderd_data[current_object][1] == 'Other':
                    item = self.model_history.item(3, 1)
                    item.setText(str(int(item.text())+1))
                self.signal_update_history_table.emit(self.model_history)

                current_object += 1

        
        print('All rubbish have been classified')
        sleep(3)
        return

# 重排物体顺序，按照厨余垃圾，有害垃圾，其他垃圾，可回收垃圾的顺序排列
    def classification_order(self, successful_data):
        # 创建列表，用于重排数据
        orderd_data = []
        # 创建列表，用于存储物体类别
        order = ['Organic','Harmful','Other','Recyclable']
        # 把successful_data中的数据按照厨余垃圾，有害垃圾，其他垃圾，可回收垃圾的顺序排列
        for i in range(4):
            for row_data in successful_data:
                if row_data[1] == order[i]:
                    orderd_data.append(row_data)
        
        # 把物体的序号重排
        for i in range(len(orderd_data)):
            orderd_data[i][0] = i+1

        # 更新表格
        self.signal_update_current_table.emit(orderd_data)
        return orderd_data

# 以下函数为挡板控制功能, db2和db4先开后关
    def board_control(self, state):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(7, GPIO.OUT)
        GPIO.setup(11, GPIO.OUT)
        GPIO.setup(13, GPIO.OUT)
        GPIO.setup(15, GPIO.OUT)

        db1=GPIO.PWM(7, 50)
        db2=GPIO.PWM(11, 50)
        db3=GPIO.PWM(13, 50)
        db4=GPIO.PWM(15, 50)             
        db1.start(0)
        db2.start(0)
        db3.start(0)
        db4.start(0)
        if state == 'open':
            print('Board open')
            db2.ChangeDutyCycle(5.5)
            db4.ChangeDutyCycle(5.1)
            sleep(0.5)
            db1.ChangeDutyCycle(5.4)
            db3.ChangeDutyCycle(4.4) 
            sleep(1)
        elif state == 'close':
            print('Board close')
            db1.ChangeDutyCycle(7.7) 
            db3.ChangeDutyCycle(6.1)
            sleep(0.5)
            db2.ChangeDutyCycle(8.0)
            db4.ChangeDutyCycle(8.2) 
            sleep(1)
        GPIO.cleanup()
        sleep(1)


# 以下函数为坐标的梯形转换函数（输入：画面坐标xy，输出梯形坐标百分比）
    def xy2trapezoid(self, x, y):
        # 输入画面坐标为(0,0)到(640,480)，原点为左上角，x轴朝右，y轴朝下
        # 梯形的上下边平行于x轴
        # 梯形四个角，左上，右上，左下，右下
        trapezoid = [[204, 59], [438, 59], [182, 262], [470, 262]]
        result_x = ((y-trapezoid[0][1])/(trapezoid[2][1]-trapezoid[0][1])*(trapezoid[0][0]-trapezoid[2][0]))/((y-trapezoid[0][1])/(trapezoid[2][1]-trapezoid[0][1])*(trapezoid[0][0]-trapezoid[2][0]+trapezoid[3][0]-trapezoid[1][0])+trapezoid[1][0]-trapezoid[0][0])
        result_y = (y-trapezoid[0][1])/(trapezoid[2][1]-trapezoid[0][1])
        return result_x, result_y

# 以下函数为单次扔垃圾抓手控制
    def arm_catch_throw(self, object_data):
        # object_data数据：[物体序号, 物体类别, 物体坐标, 物体尺寸, 置信度, 分类状态]
        # 转换后的左上角坐标
        left_top_x, left_top_y = self.xy2trapezoid(object_data[2][0], object_data[2][1])
        # 转换后的右下角坐标
        right_bottom_x, right_bottom_y = self.xy2trapezoid(object_data[2][0]+object_data[3][0], object_data[2][1]+object_data[3][1])
        # 转换后的中心坐标
        center_x = (left_top_x+right_bottom_x)/2
        center_y = (left_top_y+right_bottom_y)/2
        # 转换后的角度（与x方向的正方向夹角）（角度制）
        object_angle = math.degrees(math.atan((right_bottom_y-left_top_y)/(right_bottom_x-left_top_x)))
        # 把0到150度转换到2到12的范围，并提供一个参数，用于调节
        angle = object_angle/180*10+2
        if angle < 2:
            angle = 2
        elif angle > 12:
            angle = 12

        # 以下为机械臂移动
        arm_go_position(center_x, center_y)

        # 以下为抓取部分
        Elv_Pin = 19 #举升机
        Rot_Pin = 21 #旋转机
        Clw_Pin = 23 #爪子

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(Elv_Pin, GPIO.OUT)
        GPIO.setup(Rot_Pin, GPIO.OUT)
        GPIO.setup(Clw_Pin, GPIO.OUT)

        elv=GPIO.PWM(Elv_Pin, 50)
        rot=GPIO.PWM(Rot_Pin, 50)
        clw=GPIO.PWM(Clw_Pin, 50)
        elv.start(0)
        rot.start(0)
        clw.start(0)

        # 抓取
        print('Catch')
        clw.ChangeDutyCycle(7.8) #爪子张开
        sleep(1)
        # rot.ChangeDutyCycle(angle) #旋转到指定角度
        # sleep(1)
        elv.ChangeDutyCycle(6.2) #举升机下降
        sleep(1)

        # 捡起
        print('Pick up')
        clw.ChangeDutyCycle(6) #爪子闭合
        sleep(1)
        # rot.ChangeDutyCycle(2) #旋转角度恢复
        elv.ChangeDutyCycle(4) #举升机上升

        # 移动到指定位置垃圾桶
        if object_data[1] == 'Recyclable':
            print('Move to Recyclable')
            arm_go_position(self.arm_position_recyclable[0], self.arm_position_recyclable[1])
        elif object_data[1] == 'Organic':
            print('Move to Organic')
            arm_go_position(self.arm_position_organic[0], self.arm_position_organic[1])
        elif object_data[1] == 'Harmful':
            print('Move to Harmful')
            arm_go_position(self.arm_position_harmful[0], self.arm_position_harmful[1])
        elif object_data[1] == 'Other':
            print('Move to Other')
            arm_go_position(self.arm_position_other[0], self.arm_position_other[1])

        # 打开挡板
        print('Open board')
        self.board_control('open')

        # 松开爪子丢垃圾
        print('Throw')
        clw.ChangeDutyCycle(4.4) #爪子张开
        sleep(1)
        GPIO.cleanup()
        sleep(1)

        # 关闭挡板
        print('Close board')
        self.board_control('close')

        # 对可回收垃圾进行压缩
        if object_data[1] == 'Recyclable':
            self.press_control()

        # 满载检测
        if object_data[1] == 'Recyclable':
            self.full_detect()

        GPIO.cleanup()
        sleep(1)

# 以下函数为机械臂移动
    def arm_go_position(self, x, y):
        x_Dir_Pin = 22
        x_Pul_Pin = 24
        y_Dir_Pin = 26
        y_Pul_Pin = 32

        x_Dir = 0
        x_Steps = 0
        y_Dir = 0
        y_Steps = 0

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(x_Dir_Pin, GPIO.OUT)
        GPIO.setup(x_Pul_Pin, GPIO.OUT)
        GPIO.setup(y_Dir_Pin, GPIO.OUT)
        GPIO.setup(y_Pul_Pin, GPIO.OUT)

        # x，y方向的步进电机都是顺时针(Dir脚为1)旋转为正方向
        if x >= self.arm_current_position[0]:
            x_Dir = 1
            x_Steps = int((x-self.arm_current_position[0])*self.arm_step)
        elif x < self.arm_current_position[0]:
            x_Dir = 0
            x_Steps = int((self.arm_current_position[0]-x)*self.arm_step)

        if y >= self.arm_current_position[1]:
            y_Dir = 1
            y_Steps = int((y-self.arm_current_position[1])*self.arm_step)
        elif y < self.arm_current_position[1]:
            y_Dir = 0
            y_Steps = int((self.arm_current_position[1]-y)*self.arm_step)
        
        # 控制步进电机，x，y方向同时移动
        for i in range(max(x_Steps, y_Steps)):
            if i < x_Steps:
                GPIO.output(x_Dir_Pin, x_Dir)
                GPIO.output(x_Pul_Pin, GPIO.HIGH)
                sleep(0.001) #这几个小的sleep控制了步进电机的转速
                GPIO.output(x_Pul_Pin, GPIO.LOW)
                sleep(0.001)
            if i < y_Steps:
                GPIO.output(y_Dir_Pin, y_Dir)
                GPIO.output(y_Pul_Pin, GPIO.HIGH)
                sleep(0.001)
                GPIO.output(y_Pul_Pin, GPIO.LOW)
                sleep(0.001)
        
        sleep(2)
        # 更新当前位置
        self.arm_current_position = [x, y]
        GPIO.cleanup()
        sleep(2)


# 以下功能为载物台控制
    def platform_control(self, category):
        # 初始化GPIO库
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(10, GPIO.OUT)
        GPIO.setup(12, GPIO.OUT)
        ba=GPIO.PWM(10, 50) # Base电机，载物台底座旋转，参数范围[2，12]
        up=GPIO.PWM(12, 50) # Up电机，载物台倾倒旋转，参数范围[2，12]
        ba.start(0)
        up.start(0)
        if category =='Recyclable':
            print('Platform to Recyclable/Position 1')
            up.ChangeDutyCycle(4.5)
            ba.ChangeDutyCycle(7.5)
            sleep(2)
            # 压缩可回收垃圾
            self.press_control()
            # 满载检测
            self.full_detect()

        elif category=='Organic':
            print('Platform to Organic/Position 2')
            up.ChangeDutyCycle(8)
            ba.ChangeDutyCycle(3)
            sleep(2)
        elif category=='Harmful':
            print('Platform to Harmful/Position 3')
            up.ChangeDutyCycle(8)
            ba.ChangeDutyCycle(7.8)
            sleep(2)
        elif category=='Other':
            print('Platform to Other/Position 4')
            up.ChangeDutyCycle(4.5)
            ba.ChangeDutyCycle(4)
            sleep(2)
        print('Platform to Position 0')
        up.ChangeDutyCycle(6.4)
        ba.ChangeDutyCycle(5.2)

        GPIO.cleanup()
        sleep(1)

# 以下函数为压缩杆及其挡板控制
    def press_control(self):
        print('Press Start')
        # 设置GPIO引脚模式
        GPIO.setmode(GPIO.BOARD)
        # 定义伸缩杆GPIO引脚
        IN1 = 38
        IN2 = 36
        ENA = 40
        # 设置伸缩杆GPIO引脚为输出模式
        GPIO.setup(IN1, GPIO.OUT)
        GPIO.setup(IN2, GPIO.OUT)
        GPIO.setup(ENA, GPIO.OUT)

        # 压缩挡板舵机初始化
        GPIO.setup(8, GPIO.OUT)
        pb=GPIO.PWM(8, 50) #Press_Board
        pb.start(0)

        # 启用伸缩杆电机驱动
        GPIO.output(ENA, GPIO.HIGH)
        # 伸缩杆伸长
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        time.sleep(7)

        # 伸缩杆缩短的同时打开挡板
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        time.sleep(3)
        # 挡板打开
        pb.ChangeDutyCycle(7.5)
        time.sleep(2)
        # 挡板关闭
        pb.ChangeDutyCycle(2.5)
        time.sleep(2)
        # 停用伸缩杆电机驱动
        GPIO.output(ENA, GPIO.LOW)

        # 清理GPIO引脚设置
        GPIO.cleanup()
        print('Press End')
        sleep(1)

# 以下函数为满载检测功能实现
    def full_detect(self):
        # 设置GPIO引脚模式
        GPIO.setmode(GPIO.BOARD)

        # 定义GPIO引脚
        TRIG = 16 # 发射端
        ECHO = 18 # 接收端

        # 设置GPIO引脚为输出模式
        GPIO.setup(TRIG,GPIO.OUT)
        GPIO.setup(ECHO,GPIO.IN)
        
        # 防止信号残留
        GPIO.output(TRIG, False)
        time.sleep(0.01)

        # 发射信号
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        # 防止传感器超时无返回导致程序阻塞
        protect_time_start = time.time()
        while GPIO.input(ECHO)==0:
            pulse_start = time.time()
            if time.time() - protect_time_start > 1:
                now = datetime.datetime.now()
                time_str = now.strftime("%H:%M:%S")
                self.label_fullInfo.setText('最近检测时间：'+ str(time_str) +'\n传感器超时无返回，请检查硬件连接')
                self.label_fullState.setText('离线')
                self.label_fullState.setStyleSheet("QLabel{border-radius:40px; border:5px solid gray; color:gray;}")
                return
        while GPIO.input(ECHO)==1:
            pulse_end = time.time()
            if time.time() - protect_time_start > 1:
                now = datetime.datetime.now()
                time_str = now.strftime("%H:%M:%S")
                self.label_fullInfo.setText('最近检测时间：'+ str(time_str) +'\n传感器超时无返回，请检查硬件连接')
                self.label_fullState.setText('离线')
                self.label_fullState.setStyleSheet("QLabel{border-radius:40px; border:5px solid gray; color:gray;}")
                return
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        distance = round(distance+1.15, 2)

        now = datetime.datetime.now()
        time_str = now.strftime("%H:%M:%S")
        self.label_fullInfo.setText('最近检测时间：'+ str(time_str) +'\n传感器距离：'+ str(distance) + 'cm')
        if distance > 19 and distance < 25:
            self.label_fullState.setText('未满载')
            self.label_fullState.setStyleSheet("QLabel{border-radius:40px; border:5px solid green; color:green;}")
        else:
            self.label_fullState.setText('已满载')
            self.label_fullState.setStyleSheet("QLabel{border-radius:40px; border:5px solid red; color:red;}")

# 以下函数为播放宣传片功能实现
    def no_object_timeout(self):
        # 超时计时函数（倒数，并在时间到时启动宣传片）
        if self.timeout_count != 0:
            self.label_countdown.setText('宣传片超时播放剩余：' + str(self.timeout_count))
            self.timeout_count = round(self.timeout_count - 0.1, 1)
        else:
            self.label_countdown.setText('宣传片超时播放剩余：0.0')
            self.film_on = True
            self.timer_play_film.start()
            self.label_countdown.setVisible(False)
            self.timeout_count = self.timeout_default_count
            self.timer_timeout.stop()

    def play_film(self):
        success, img0 = self.cap_film.read()
        if success:
            # 显示视频帧
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            height, width, channel = img0.shape
            bytesPerLine = 3 * width
            qImg = QImage(img0.data, width, height,
                        bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.label_videoStream.setPixmap(pixmap)
        else:
            # 检查是否播放到末尾
            if self.cap_film.get(cv2.CAP_PROP_POS_FRAMES) == self.cap_film.get(cv2.CAP_PROP_FRAME_COUNT):
                # 重新开始播放
                self.cap_film.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                # 视频读取失败
                print("Error reading video file")

# 以下函数为了响应信号，不能在子线程中直接调用ui的控件或者启停QTimer，否则会出bug/报错
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
    
    def update_history_table(self, data):
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

    def control_timer_play_film(self, flag):
        if flag:
            self.timer_play_film.start()
            self.label_countdown.setVisible(False)
        else:
            self.timer_play_film.stop()
            self.label_countdown.setVisible(True)

    def control_timer_timeout(self, flag):
        if flag:
            self.timer_timeout.start()
        else:
            self.timer_timeout.stop()

#以下函数重写了父类函数
    def resizeEvent(self, event: QResizeEvent) -> None:
        # 计算widget_video比例，使得画面可以自适应的同时保持锁定4:3比例
        ratio = 4 / 3
        w = self.widget_video.width()
        h = self.widget_video.height()
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
        # 调整倒计时的位置为右下角
        self.label_countdown.setGeometry(self.label_videoStream.x()+self.label_videoStream.width()-self.label_countdown.width(),
                                            self.label_videoStream.y()+self.label_videoStream.height()-self.label_countdown.height(),
                                            self.label_countdown.width(),
                                            self.label_countdown.height())
        return super().resizeEvent(event)

    def closeEvent(self, event):
        # 停止宣传片相关的计时器
        self.timer_play_film.stop()
        self.timer_timeout.stop()
        # 清理GPIO
        if platform == 'Rasp':
            GPIO.cleanup()
        # 停止detect函数
        self.run_flag = False
        self.cap.release()
        # 退出程序
        super().closeEvent(event)

if __name__ == "__main__":

    # 模型加载
    model_pb_path = "best.onnx"
    so = ort.SessionOptions()
    net = ort.InferenceSession(model_pb_path, so)
    # 标签字典
    dic_labels = {0: 'Organic', 1: 'Recyclable', 2: 'Recyclable', 3: 'Other',4:'Harmful',5:'Harmful',
    6:'Recyclable',7:'Other',8:'Organic',9:'Organic'}
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
