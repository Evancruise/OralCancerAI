# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:39:28 2021

@author: USER
# pyinstaller -p utils -p model_resnet -p visualize -w main_debug_v2.py
"""

import sys
import cv2 as cv
import unicorn

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import json
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication
from PyQt5.QtWidgets import * 
from imantics import Mask
from shapely.geometry.polygon import Polygon

#from datetime import datetime
#from config import Config
#import utils
#import model_resnet as modellib
#import visualize
#from mrcnn import visualize

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 641)
        self._window = MainWindow
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnInput = QtWidgets.QPushButton(self.centralwidget)
        self.btnInput.setGeometry(QtCore.QRect(100, 330, 93, 28))
        self.btnInput.setObjectName("btnInput")
        self.btnTest = QtWidgets.QPushButton(self.centralwidget)
        self.btnTest.setGeometry(QtCore.QRect(290, 330, 93, 28))
        self.btnTest.setObjectName("btnTest")
        
        self.btnTest_total_clicked = QtWidgets.QPushButton(self.centralwidget)
        self.btnTest_total_clicked.setGeometry(QtCore.QRect(290, 360, 103, 28))
        self.btnTest_total_clicked.setObjectName("btnTest")
        
        self.btnSave = QtWidgets.QPushButton(self.centralwidget)
        self.btnSave.setGeometry(QtCore.QRect(480, 330, 93, 28))
        self.btnSave.setObjectName("btnSave")
        self.btnmda = QtWidgets.QPushButton(self.centralwidget)
        self.btnmda.setGeometry(QtCore.QRect(760, 330, 93, 28))
        self.btnmda.setObjectName("btnmda")
        
        self.labelinput = QtWidgets.QLabel(self.centralwidget)
        self.labelinput.setGeometry(QtCore.QRect(50, 40, 291, 281))
        self.labelinput.setObjectName("labelinput")
        self.labelresult = QtWidgets.QLabel(self.centralwidget)
        self.labelresult.setGeometry(QtCore.QRect(370, 40, 311, 281))
        self.labelresult.setObjectName("labelresult")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(50, 390, 291, 192))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.btnInput.clicked.connect(MainWindow.btnInput_Clicked)
        self.btnTest.clicked.connect(MainWindow.btnTest_Clicked)
        self.btnTest.pressed.connect(MainWindow.btnTest_Pressed)
        self.btnTest_total_clicked.clicked.connect(MainWindow.btnTest_total_Clicked)
        self.btnSave.clicked.connect(MainWindow.btnSave_Clicked)
        self.btnmda.clicked.connect(MainWindow.btnmad_Clicked)
        
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        list_widget, list_widget_label = self.UiComponents()
        return list_widget, list_widget_label
    
    def UiComponents(self):
        list_widget = QListWidget(self)
        list_widget_label = QListWidget(self)
        list_widget.setGeometry(350, 390, 291, 192)
        list_widget_label.setGeometry(650, 390, 291, 192)
        
        scroll_bar = QScrollBar(self)
        scroll_bar_label = QScrollBar(self)
        scroll_bar.setStyleSheet("background : lightgreen;")
        scroll_bar_label.setStyleSheet("background : lightgreen;")
        list_widget.setVerticalScrollBar(scroll_bar)
        list_widget_label.setVerticalScrollBar(scroll_bar_label)
        value = list_widget.verticalScrollBar()
        value2 = list_widget_label.verticalScrollBar()
        return list_widget, list_widget_label
        
    def retranslateUi(self, MainWindow):
        self._translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(self._translate("MainWindow", "MainWindow"))
        self.btnInput.setText(self._translate("MainWindow", "import image"))
        self.btnTest.setText(self._translate("MainWindow", "inference"))
        self.btnTest_total_clicked.setText(self._translate("MainWindow", "inference (total)"))
        self.btnSave.setText(self._translate("MainWindow", "save image"))
        self.btnmda.setText(self._translate("MainWindow", "modify image"))
        self.labelinput.setText(self._translate("MainWindow", "input image"))
        self.labelresult.setText(self._translate("MainWindow", "results image"))

class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.list_widget, self.list_widget_label = self.setupUi(self)
        self.setWindowTitle("oral cancer autodetection UI platform")
        self.filenames = []
        self.index_i = -1
        self.annotation_index = {}

        self.labelinput.setAlignment(Qt.AlignCenter)
        self.labelinput.setStyleSheet("QLabel{background:gray;}"
                                      "QLabel{color:rgba(255,255,255,150);"
                                      "font-size:20px;"
                                      "font-weight:bold;"
                                      "font-family:Roman times;}")

        self.labelresult.setAlignment(Qt.AlignCenter)
        self.labelresult.setStyleSheet("QLabel{background:gray;}"
                                       "QLabel{color:rgba(255,255,255,150);"
                                       "font-size:20px;"
                                       "font-weight:bold;"
                                       "font-family:Roman times;}")
    
    def btnTest_Pressed(self):
        if not hasattr(self, "captured"):
            return
        self.textBrowser.append("Image inferencing...")
    
    def btnInput_Clicked(self):
        filename, filename_direc = QFileDialog.getOpenFileName(self, 'Open image', "", "*.png;;*.jpg;;All Files(*)")
        
        self.files_root = ''
        
        for i in filename.split('/')[:-1]:
            self.files_root = self.files_root + i + '/'
        
        filename = glob.glob(filename.split('00')[0] + '/*.png')
        
        self.textBrowser.setPlainText("Open image success")
        self.add_image_names(filename)
    
    def add_image_names(self, fname):
        
        self.image_info = {"Run": False, 'dir': "", 'index': "", 'prev name': "", 'next name': "", 'name': "", 'labels': [], 'colors':[], 'file': "", 'segmentations': [], 'xpoints': [], 'ypoints': [], 'i': 0}
        
        self.list_widget.clear()
        
        self.filenames = []
        
        for i in fname:
            item = QListWidgetItem(i)
            self.list_widget.addItem(item)
            self.filenames.append(i)
        
        self.list_widget.itemDoubleClicked.connect(self.onClicked)
    
    def onClicked(self, item):
        filename = item.text()
        
        self.labelresult.setText(self._translate("MainWindow", "results image"))
        
        self.image_info["name"] = str(item.text())
        
        self.captured = cv.imread(self.image_info["name"])
        self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)
        self.captured = cv.resize(self.captured, (512, 384), interpolation=cv.INTER_AREA)
        
        rows, cols, channels = self.captured.shape
        
        bytesPerLine = channels * cols
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelinput.setPixmap(QPixmap.fromImage(QImg).scaled(
        self.labelinput.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        if os.path.exists('temp/' + self.image_info["name"].split('\\')[-1]):
            self.captured = cv.imread('temp/' + self.image_info["name"].split('\\')[-1])
            self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)
            self.captured = cv.resize(self.captured, (512, 384), interpolation=cv.INTER_AREA)

            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.index_i = -1
        
        if not os.path.exists('json_test/lesion_oralCa_20200319~20200924_{}_merge.json'.format(filename.split('/')[-1].split('\\')[-1].split('_')[0])):
            QMessageBox.information(self,"Hint","File not exist")
            return 
        else:
            with open('json_test/lesion_oralCa_20200319~20200924_{}_merge.json'.format(filename.split('/')[-1].split('\\')[-1].split('_')[0]), encoding='utf-8', errors='ignore') as json_file:
                predicted_information = json.loads(json_file.read())
        
            self.image_info = {}
        
            self.labelresult.setText(self._translate("MainWindow", "results image"))
        
            self.image_info["name"] = str(item.text())
            self.image_info['labels'] = []
            self.image_info['segmentations'] = []
            self.image_info['xpoints'] = []
            self.image_info['ypoints'] = []
        
            for i in range(len(predicted_information['regions'][0])):
                category = predicted_information['regions'][0]['region_{}'.format(i+1)][0]["category"]
                if len(predicted_information['regions'][0]['region_{}'.format(i+1)][0]["bbox"]) == 0:
                    break
                bbox = predicted_information['regions'][0]['region_{}'.format(i+1)][0]["bbox"][0]
                
                all_points_x = predicted_information['regions'][0]['region_{}'.format(i+1)][0]["all_points_x"]
                all_points_y = predicted_information['regions'][0]['region_{}'.format(i+1)][0]["all_points_y"]
                segmenation = predicted_information['regions'][0]['region_{}'.format(i+1)][0]["segmentation"]
                self.image_info['labels'].append([i, category, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                self.image_info['segmentations'].append(segmenation)
                self.image_info['xpoints'].append(all_points_x)
                self.image_info['ypoints'].append(all_points_y)
        
        self.captured = cv.imread('temp/' + self.image_info["name"].split('\\')[-1])
        self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)
        #self.captured = cv.resize(self.captured, (512, 384), interpolation=cv.INTER_AREA)

        rows, cols, channels = self.captured.shape
        bytesPerLine = channels * cols
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.add_label_information()
    
    def get_ax(self, rows=1, cols=1, size=8):
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
    
    def one_hot(self, mask):
        one_hot = np.zeros((mask.shape[0], mask.shape[1]))
    
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j] > 0:
                    one_hot[i,j] = 255
                
        return one_hot
    
    def create_polygon_and_bbox(self, mask):
        sub_mask_convert = self.one_hot(mask)
        polygon = Mask(sub_mask_convert).polygons()
    
        all_xpoints = []
        all_ypoints = []
        segmentation = []
        bbox = []
    
        if len(polygon.points[0]) != 0:
            for i in range(len(polygon.points[0])):
                all_xpoints.append(polygon.points[0][i][0]) 
                segmentation.append(polygon.points[0][i][0])
                all_ypoints.append(polygon.points[0][i][1])
                segmentation.append(polygon.points[0][i][1])
            bbox.append(np.min(all_ypoints))
            bbox.append(np.min(all_xpoints)) 
            bbox.append(np.max(all_ypoints)-np.min(all_ypoints)) 
            bbox.append(np.max(all_xpoints)-np.min(all_xpoints)) 
        else:
            all_xpoints = []
            all_ypoints = []
            segmentation = []
            bbox = []
    
        return bbox, segmentation, all_xpoints, all_ypoints
    
    def btnTest_Clicked(self):
        
        if not hasattr(self, "captured"):
            self.textBrowser.setPlainText("No image input")
            return
        
        if not (os.path.exists('mrcnn/config.py') or os.path.exists('mrcnn/utils.py') or os.path.exists('mrcnn/model_resnet.py') or os.path.exists('mrcnn/visualize.py')):
            QMessageBox.warning(self,"Hint","Please put the config, utils, model_resnet and visualize documents first.")
            return 
            
        from mrcnn.config import Config
        import mrcnn.utils as utils
        import mrcnn.model_resnet as modellib
        import mrcnn.visualize as visualize
        
        class ShapesConfig(Config):
            # Give the configuration a recognizable name
            NAME = "shapes"

            category_name = ['G', 'Y', 'R']
            colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]
            colorlist = ['Gray', 'Green', 'Yellow', 'Red']

            # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
            # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

            # Number of classes (including background)
            NUM_CLASSES = 1 + 3  # background + 3 shapes
            
            BACKBONE = "resnet101"

            # Use small images for faster training. Set the limits of the small side
            # the large side, and that determines the image shape.
            IMAGE_MIN_DIM = 320
            IMAGE_MAX_DIM = 512

            # Use smaller anchors because our image and objects are small
            RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

            # Reduce training ROIs per image because the images are small and have
            # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
            TRAIN_ROIS_PER_IMAGE = 100

            # Use a small epoch since the data is simple
            STEPS_PER_EPOCH = 100

            # use small validation steps since the epoch is small
            VALIDATION_STEPS = 50
            
            MEAN_PIXEL_LOL = np.array([90, 91, 70, 141])
            VARIANCE_LOL = np.array([3587, 3146, 2022, 6594])
            
            USE_MINI_MASK = True
    
            MINI_MASK_SHAPE = (28, 28)
            IMAGE_RESIZE_MODE = "square"
            IMAGE_MIN_DIM = 856
            IMAGE_MAX_DIM = 1024
    
            IMAGE_MIN_SCALE = 0
            IMAGE_CHANNEL_COUNT = 3
    
            N_CHANNELS = IMAGE_CHANNEL_COUNT
            MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

        # import train_tongue
        # class InferenceConfig(coco.CocoConfig):
        class InferenceConfig(ShapesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        colors = config.colors
        
        ROOT_DIR = os.getcwd()

        # Import Mask RCNN
        #sys.path.append(ROOT_DIR)  # To find local version of the library
        #from mrcnn import utils
        #import mrcnn.model_resnet as modellib
        #from mrcnn import visualize

        # Import COCO config
        # sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
        # from samples.coco import coco
        
        if not os.path.exists('logs'):
            QMessageBox.warning(self,"Hint","Please put the log files first for testing oral images.") 
            return

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights("./logs/microcontroller_detection20211014T2052_resnet152/mask_rcnn_resnet152_microcontroller_detection_0200.h5", by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        class_names = ['BG', 'G', 'Y', 'R']
        # Load a random image from the images folder
        # file_names = next(os.walk(IMAGE_DIR))[2]
        image = cv.imread(self.image_info["name"])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (512, 384), interpolation=cv.INTER_AREA)

        #print('image.shape:', image.shape)
        
        #a = datetime.now()
        # Run detection
        results = model.detect([image], verbose=1)
        
        self.annotation_index = {}
        
        # print("result:",results)
        #b = datetime.now()
        # Visualize results
        #time = (b - a).seconds
        #self.textBrowser.append("Elapsed time: " + str(time) + "s")
        self.r = results[0]

        color_list = []
        cls_index = []
        
        for m, cls_name in enumerate(self.r['class_ids']):
            if cls_name == 3:
                color_list.append(colors[2])
            if cls_name == 2:
                color_list.append(colors[1])
            if cls_name == 1:
                color_list.append(colors[0])
            cls_index.append(cls_name)
            
        #image_display, mask_display = visualize.display_instances_auto_label(image, r['rois'], r['masks'], r['class_ids'], 
        #                    ['B', 'G', 'Y', 'R'], r['scores'], ax=self.get_ax(), colors=color_list)
        image_display = visualize.display_instances(image, self.r['rois'], self.r['masks'], self.r['class_ids'], 
                            ['B', 'G', 'Y', 'R'], self.r['scores'], ax=self.get_ax(), colors=color_list)
        
        #print("self.image_info['name']:", self.image_info["name"])
        if not os.path.exists('temp/'):
            os.mkdir('temp/')
        
        plt.imshow(image_display)
        plt.savefig('temp/' + self.image_info["name"].split('\\')[-1], bbox_inches='tight', pad_inches=0)
        plt.close()
        
        class_ids = self.r['class_ids']

        class_dict = {}
        for index, i in enumerate(class_ids):
            if i in class_dict:
                class_dict[i] += 1
            else:
                class_dict[i] = 1

        output_dict = {}

        for key, value in class_dict.items():
            label = class_names[key]
            output_dict[label] = value
        self.textBrowser.append("Included images:")
        for key, value in output_dict.items():
            self.textBrowser.append(str(value) + "" + str(key))
	
        #print(image_display.shape)
        self.captured = cv.imread('temp/' + self.image_info["name"].split('\\')[-1])
        self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)
        #self.captured = cv.resize(self.captured, (512, 384), interpolation=cv.INTER_AREA)

        rows, cols, channels = self.captured.shape
        bytesPerLine = channels * cols
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.textBrowser.append("Inference complete.")
        
        self.image_info['labels'] = []
        self.image_info['segmentations'] = []
        self.image_info['xpoints'] = []
        self.image_info['ypoints'] = []
        
        i = 0
        
        for class_id, rois_id, mask_id, segmentations_id in zip(self.r['class_ids'], self.r['rois'], self.r['masks'], self.r['segmentations']):
        
            bbox_coor, polygon_coor, ploygon_x, ploygon_y = self.create_polygon_and_bbox(segmentations_id)
        
            if int(class_id) == 1:
                #label_information.append([i, 'G', rois_id[0], rois_id[1], rois_id[2], rois_id[3]])
                self.image_info['labels'].append([i, 'G', rois_id[0], rois_id[1], rois_id[2], rois_id[3]])
                self.image_info['segmentations'].append(polygon_coor)
                self.image_info['xpoints'].append(ploygon_x)
                self.image_info['ypoints'].append(ploygon_y)
            elif int(class_id) == 2:
                #label_information.append([i, 'Y', rois_id[0], rois_id[1], rois_id[2], rois_id[3]])
                self.image_info['labels'].append([i, 'Y', rois_id[0], rois_id[1], rois_id[2], rois_id[3]])
                self.image_info['segmentations'].append(polygon_coor)
                self.image_info['xpoints'].append(ploygon_x)
                self.image_info['ypoints'].append(ploygon_y)
            elif int(class_id) == 3:
                #label_information.append([i, 'R', rois_id[0], rois_id[1], rois_id[2], rois_id[3]])
                self.image_info['labels'].append([i, 'R', rois_id[0], rois_id[1], rois_id[2], rois_id[3]])
                self.image_info['segmentations'].append(polygon_coor)
                self.image_info['xpoints'].append(ploygon_x)
                self.image_info['ypoints'].append(ploygon_y)
            i += 1
        
        self.add_label_information()
        
        self.display_image()
    
    def btnTest_total_Clicked(self):
        
        num = len(self.filenames)
        progress = QProgressDialog(self)
        progress.setWindowTitle("Waiting")  
        progress.setLabelText("Loading...")
        progress.setCancelButtonText("Cancel")
        progress.setMinimumDuration(2)
        progress.setWindowModality(Qt.WindowModal)
        progress.setRange(0,num) 

        class ShapesConfig(Config):
            NAME = "shapes"

            category_name = ['G', 'Y', 'R']
            colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]
            colorlist = ['Gray', 'Green', 'Yellow', 'Red']
            
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

            NUM_CLASSES = 1 + 3
            
            BACKBONE = "resnet101"

            IMAGE_MIN_DIM = 320
            IMAGE_MAX_DIM = 512

            RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

            TRAIN_ROIS_PER_IMAGE = 100

            STEPS_PER_EPOCH = 100

            VALIDATION_STEPS = 50
            
            MEAN_PIXEL_LOL = np.array([90, 91, 70, 141])
            VARIANCE_LOL = np.array([3587, 3146, 2022, 6594])
            
            USE_MINI_MASK = True
    
            MINI_MASK_SHAPE = (28, 28)
            IMAGE_RESIZE_MODE = "square"
            IMAGE_MIN_DIM = 856
            IMAGE_MAX_DIM = 1024
    
            IMAGE_MIN_SCALE = 0
            IMAGE_CHANNEL_COUNT = 3
    
            N_CHANNELS = IMAGE_CHANNEL_COUNT
            MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
        
        class InferenceConfig(ShapesConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        
        config = InferenceConfig()
        colors = config.colors
        
        ROOT_DIR = os.getcwd()

        # Import Mask RCNN
        #sys.path.append(ROOT_DIR)  # To find local version of the library
        #from mrcnn import utils
        #import mrcnn.model_resnet as modellib
        #from mrcnn import visualize

        # Import COCO config
        # sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
        # from samples.coco import coco

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        
        model.load_weights("./logs/microcontroller_detection20211014T2052_resnet152/mask_rcnn_resnet152_microcontroller_detection_0200.h5", by_name=True)
        
        class_names = ['BG', 'G', 'Y', 'R']
        
        for i in range(num):
            
            progress.setValue(i+1) 
            
            if progress.wasCanceled():
                QMessageBox.warning(self,"Hint","Inference terminates.") 
                break
            
            image = cv.imread(self.filenames[i])
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            rows, cols, channels = image.shape
            bytesPerLine = channels * cols
            QImg = QImage(image.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelinput.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelinput.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            obj={}
            all_xpoints = []
            all_ypoints = []
            segmentation = []
            bbox = []
            
            results = model.detect([image], verbose=1)
            
            r = results[0]
            
            color_list = []
            cls_index = []
            
            for m, cls_name in enumerate(r['class_ids']):
                if cls_name == 3:
                    color_list.append(colors[2])
                if cls_name == 2:
                    color_list.append(colors[1])
                if cls_name == 1:
                    color_list.append(colors[0])
                cls_index.append(cls_name)
            
            j = 1
            
            if len(r['class_ids']) == 0:
                obj["region_1"] = []
                obj["region_1"].append({
                    "category": 'G',
                    "name": "polygon",
                    "all_points_x": [],
                    "all_points_y": [],
                    "segmentation": [],
                    "bbox": [],
                    "region_attributes":{}
                })
                
                annotation_index = {}
                annotation_index["fileref"] = []
                annotation_index["size"] = 199608
                annotation_index["filename"] = 'train/all_crop_v2/' + self.filenames[i].split('/')[-1].split('\\')[-1]
                annotation_index["category_id"] = 'G'
                annotation_index["base64_img_data"] = []
                annotation_index["file_attributes"] = []
                annotation_index["regions"] = []
                annotation_index["regions"].append(obj)
            
            else:
                
                class_id_all = []
                
                for class_id, rois_id, mask_id, segmentations_id in zip(r['class_ids'], r['rois'], r['masks'], r['segmentations']):
                    
                    all_xpoints = []
                    all_ypoints = []
                    segmentation = []
                    bbox = []
            
                    bbox_coor, polygon_coor, ploygon_x, ploygon_y = self.create_polygon_and_bbox(segmentations_id)
                
                    for x_point_id, y_point_id in zip(ploygon_x, ploygon_y):
                        all_xpoints.append(x_point_id) 
                        all_ypoints.append(y_point_id)
                        segmentation.append(x_point_id)
                        segmentation.append(y_point_id)
                    
                    class_id_all.append(class_id)
                
                    bbox.append([np.min(ploygon_y), np.min(ploygon_x), np.max(ploygon_y)-np.min(ploygon_y),np.max(ploygon_x)-np.min(ploygon_x)])
                
                    obj["region_{}".format(j)] = []
                    obj["region_{}".format(j)].append({
                        "category": class_names[class_id],
                        "name": "polygon",
                        "all_points_x": np.asarray(all_xpoints).tolist(),
                        "all_points_y": np.asarray(all_ypoints).tolist(),
                        "segmentation": np.asarray(segmentation).tolist(),
                        "bbox": np.asarray(bbox).tolist(),
                        "region_attributes":{}
                    })
                
                    j += 1
                    
                annotation_index = {}
                annotation_index["fileref"] = []
                annotation_index["size"] = 196608
            
                annotation_index["filename"] = 'train/all_crop_v2/' + self.filenames[i].split('/')[-1].split('\\')[-1]
                if len(class_names) > 0:
                    annotation_index["category_id"] = str(class_names[np.max(class_id_all)])
                else:
                    annotation_index["category_id"] = 'G'
                annotation_index["base64_img_data"] = []
                annotation_index["file_attributes"] = []
                annotation_index["regions"] = []
                annotation_index["regions"].append(obj)
            
            if not os.path.exists('json_test/'):
                os.mkdir('json_test/')
                
            with open('json_test/lesion_oralCa_20200319~20200924_{}_merge.json'.format(self.filenames[i].split('/')[-1].split('\\')[-1].split('_')[0]), 'w') as json_file:
                json.dump(annotation_index, json_file)
            
            image_display = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            ['B', 'G', 'Y', 'R'], r['scores'], ax=self.get_ax(), colors=color_list)
        
            plt.imshow(image_display)
            plt.savefig('temp/' + self.filenames[i].split('/')[-1].split('\\')[-1], bbox_inches='tight', pad_inches=0)
            plt.close()
            
            self.captured = cv.imread('temp/' + self.filenames[i].split('/')[-1].split('\\')[-1])
            self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)
            #self.captured = cv.resize(self.captured, (512, 384), interpolation=cv.INTER_AREA)

            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            progress.setValue(num)            
            QMessageBox.information(self,"Hint","Inference complete.")
    
    def mouse_handler(self, event, x, y, flags, data):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(data['img'], (x,y), 3, (0,0,255), 5, 16) 
            cv.imshow("Image", data['img'])
            data['xpoints'].append(x)
            data['ypoints'].append(y)
            data['bbox'] = [np.min(data['ypoints']), np.min(data['xpoints']), np.max(data['ypoints'])-np.min(data['ypoints']),np.max(data['xpoints'])-np.min(data['xpoints'])]
    
    def label_update_append(self, action, class_index):
        name = self.image_info['name']
        img = cv.imread(name)
    
        data = {}
        data['img'] = img.copy()
        data['xpoints'] = []
        data['ypoints'] = []
        data['bbox'] = []
        
        cv.namedWindow("Image", 0)
    
        h, w, dim = img.shape
        cv.resizeWindow("Image", w, h)
        
        i=0

        data['img'] = img.copy()
        cv.imshow('Image',data['img'])
        cv.setMouseCallback("Image", self.mouse_handler, data)
    
        while True:
            cv.imshow("Image", data['img'])
        
            k = cv.waitKey(delay=1) & 0xFF
            
            if k == 113 or k == 81:
                if len(data['xpoints']) > 0:
                    if action == 'Append':
                        self.image_info['labels'].append([int(len(self.image_info['labels'])), str(class_index), int(np.min(data['ypoints'])), int(np.min(data['xpoints'])), int(np.max(data['ypoints'])), int(np.max(data['xpoints']))])
                        break
                    elif action == 'Modify':
                        self.image_info['labels'].pop(int(self.index_i-1))
                        self.image_info['segmentations'].pop(int(self.index_i-1))
                        self.image_info['xpoints'].pop(int(self.index_i-1))
                        self.image_info['ypoints'].pop(int(self.index_i-1))
                    
                        self.image_info['labels'].append([int(len(self.image_info['labels'])), str(class_index), int(np.min(data['ypoints'])), int(np.min(data['xpoints'])), int(np.max(data['ypoints'])), int(np.max(data['xpoints']))])
                        break
    
        cv.destroyAllWindows()
    
        self.image_info['segmentations'].append(data['xpoints'])
        self.image_info['segmentations'].append(data['ypoints'])
        self.image_info['xpoints'].append(data['xpoints'])
        self.image_info['ypoints'].append(data['ypoints'])
    
        self.image_update()
    
    def image_update(self):
        
        self.list_widget_label.clear()
        self.obj = {}
        
        list_ = '0' + ' ' +  'None' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + '0'
        item = QListWidgetItem(list_)
        self.list_widget_label.addItem(item)
        
        for i in range(len(self.image_info['labels'])):
            info = self.image_info['labels'][i]
            list_ = str(i+1) + ' ' +  str(info[1]) + ' ' + str(info[2]) + ' ' + str(info[3]) + ' ' + str(info[4]) + ' ' + str(info[5])
            item = QListWidgetItem(list_)
            self.list_widget_label.addItem(item)
        
        pic = cv.imread(self.files_root + self.image_info['name'].split('\\')[-1])
        pic = cv.cvtColor(pic, cv.COLOR_BGR2RGB)
        pic = cv.resize(pic, (512, 384), interpolation=cv.INTER_AREA)
        
        fig,ax = plt.subplots(1)
        
        ax.imshow(pic)
        
        colors = ['G', 'Y', 'R']
        color_index_total = []
        
        for i in range(len(self.image_info['labels'])):
            self.obj["region_{}".format(i)] = []
            segmentation = []
            segmentation_sort_into_file = []
            all_xpoints = []
            all_ypoints = []
            for x_point_id, y_point_id in zip(self.image_info['xpoints'][i], self.image_info['ypoints'][i]):
                all_xpoints.append(x_point_id) 
                all_ypoints.append(y_point_id)
                segmentation.append((x_point_id,y_point_id))
                segmentation_sort_into_file.append(x_point_id)
                segmentation_sort_into_file.append(y_point_id)
            
            bbox = [int(self.image_info['labels'][i][3]), int(self.image_info['labels'][i][2]), int(self.image_info['labels'][i][5]-int(self.image_info['labels'][i][3])), int(self.image_info['labels'][i][4])-int(self.image_info['labels'][i][2])]
            if str(self.image_info['labels'][i][1]) == 'R':
                color = 'red'
                color_index = 2
                color_index_total.append(2)
            elif str(self.image_info['labels'][i][1]) == 'Y':
                color = 'yellow'
                color_index = 1
                color_index_total.append(1)
            elif str(self.image_info['labels'][i][1]) == 'G':
                color = 'green'
                color_index = 0
                color_index_total.append(0)
            self.obj["region_{}".format(i)].append({
                "category": colors[color_index],
                "name": "polygon",
                "all_points_x": np.asarray(all_xpoints).tolist(),
                "all_points_y": np.asarray(all_ypoints).tolist(),
                "segmentation": np.asarray(segmentation_sort_into_file).tolist(),
                "bbox": np.asarray(bbox).tolist(),
                "region_attributes":{}
            })
            
            rect = patches.Rectangle((self.image_info['labels'][i][3], self.image_info['labels'][i][2]), self.image_info['labels'][i][5]-self.image_info['labels'][i][3], self.image_info['labels'][i][4]-self.image_info['labels'][i][2], edgecolor = color, fill=False)
            rect.set_linestyle('dashed')
            
            ax.add_patch(rect)
                
            polygon = patches.Polygon(segmentation, edgecolor = color, fill=False)
                
            ax.add_patch(polygon)
            
            ax.text(int(self.image_info['labels'][i][3]), int(self.image_info['labels'][i][2]) + 8, str(self.image_info['labels'][i][1]), color='w', size=10, backgroundcolor="none")
        
        ax.set_axis_off()
        plt.savefig('temp/' + self.image_info['name'].split('\\')[-1], bbox_inches='tight', pad_inches=0)
        
        self.annotation_index = {}
        self.annotation_index["fileref"] = []
        self.annotation_index["size"] = 512 * 384
        self.annotation_index["filename"] = 'train/all_crop_v2/' + self.image_info['name'].split('\\')[-1]
        if len(color_index_total) > 1:
            self.annotation_index["category_id"] = colors[np.max(color_index_total)]
        else:
            self.annotation_index["category_id"] = 'G'
        self.annotation_index["base64_img_data"] = []
        self.annotation_index["file_attributes"] = []
        self.annotation_index["regions"] = []
        self.annotation_index["regions"].append(self.obj)
          
        self.captured_modified_ = cv.imread("temp/" + self.image_info["name"].split('\\')[-1])
        self.captured_modified_ = cv.cvtColor(self.captured_modified_, cv.COLOR_BGR2RGB)
        #self.captured_modified_ = cv.resize(self.captured_modified_, (512, 384), interpolation=cv.INTER_AREA)

        rows, cols, channels = self.captured_modified_.shape
        bytesPerLine = channels * cols
        QImg = QImage(self.captured_modified_.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def btnmad_Clicked(self):
        
        if self.index_i == -1:
            QMessageBox.information(self, 'Hint', 'Please select an image name.')
            return
        
        ret = QMessageBox.question(self, 'Want to modify the label?\n MessageBox', "Yes (Append), OK (Modify), No (Delete)", QMessageBox.Yes | QMessageBox.No | QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
        
        if ret == QMessageBox.Yes:
            class_index = QMessageBox.question(self, 'Want to modify the label?\n MessageBox', "Yes (G), Ok (Y), No (R)", QMessageBox.Yes | QMessageBox.No | QMessageBox.Ok, QMessageBox.Ok)
            if class_index == QMessageBox.Yes:
                class_i = 'G'
            elif class_index == QMessageBox.Ok:
                class_i = 'Y'
            elif class_index == QMessageBox.No:
                class_i = 'R'    
            self.label_update_append('Append', class_i)
            
        elif ret == QMessageBox.No:
            if self.index_i == 0:
                QMessageBox.information(self, 'Hint', 'Please select positive index.')
                return
            
            if len(self.image_info['labels']) == 0:
                QMessageBox.information(self, 'Hint', 'The inference result contains no any lesions.')
                return
            
            self.image_info['labels'].pop(int(self.index_i-1))
            self.image_info['xpoints'].pop(int(self.index_i-1))
            self.image_info['ypoints'].pop(int(self.index_i-1))
            self.image_update()
        
        elif ret == QMessageBox.Ok:
            if self.index_i == 0:
                QMessageBox.information(self, 'Hint', 'Please select positive index.')
                return
            
            if len(self.image_info['labels']) == 0:
                QMessageBox.information(self, 'Hint', 'The inference result contains no any lesions.')
                return
            
            class_index = QMessageBox.question(self, 'Want to modify the label?\n MessageBox', "Yes (G), Ok (Y), No (R)", QMessageBox.Yes | QMessageBox.No | QMessageBox.Ok, QMessageBox.Ok)
            
            if class_index == QMessageBox.Yes:
                class_i = 'G'
            elif class_index == QMessageBox.Ok:
                class_i = 'Y'
            elif class_index == QMessageBox.No:
                class_i = 'R'    
            self.label_update_append('Modify', class_i)
    
    def add_label_information(self):
        
        self.list_widget_label.clear()
        
        list_ = '0' + ' ' +  'None' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + '0'
        item = QListWidgetItem(list_)
        self.list_widget_label.addItem(item)
        
        for info in self.image_info['labels']:
            list_ = str(info[0]+1) + ' ' +  str(info[1]) + ' ' + str(info[2]) + ' ' + str(info[3]) + ' ' + str(info[4]) + ' ' + str(info[5])
            item = QListWidgetItem(list_)
            self.list_widget_label.addItem(item)
        
        self.list_widget_label.itemDoubleClicked.connect(self.onClicked2)
    
    def display_image(self):
        
        pic = cv.cvtColor(cv.imread(self.files_root + self.image_info["name"].split('\\')[-1]), cv.COLOR_BGR2RGB)
        pic = cv.resize(pic, (512, 384), interpolation=cv.INTER_AREA)
        
        self.obj = {}
    
        fig,ax = plt.subplots(1)
        
        ax.imshow(pic)
        
        color_index_total = []
        color_index_total.append(0)
        colors = ['G', 'Y', 'R']
        
        for i in range(len(self.image_info['labels'])):
            segmentation = []
            segmentation_sort_into_file = []
            all_xpoints = []
            all_ypoints = []
            
            self.obj["region_{}".format(i)] = []
            
            for x_point_id, y_point_id in zip(self.image_info['xpoints'][i], self.image_info['ypoints'][i]):
                all_xpoints.append(x_point_id) 
                all_ypoints.append(y_point_id)
                segmentation.append((x_point_id,y_point_id))
                segmentation_sort_into_file.append(x_point_id)
                segmentation_sort_into_file.append(y_point_id)
                
            bbox = [int(self.image_info['labels'][i][3]), int(self.image_info['labels'][i][2]), int(self.image_info['labels'][i][5]-int(self.image_info['labels'][i][3])), int(self.image_info['labels'][i][4])-int(self.image_info['labels'][i][2])]
            
            if str(self.image_info['labels'][i][1]) == 'R':
                color = 'red'
                color_index = 2
                color_index_total.append(2)
            elif str(self.image_info['labels'][i][1]) == 'Y':
                color = 'yellow'
                color_index = 1
                color_index_total.append(1)
            elif str(self.image_info['labels'][i][1]) == 'G':
                color = 'green'
                color_index = 0
                color_index_total.append(0)
            
            rect = patches.Rectangle((self.image_info['labels'][i][3], self.image_info['labels'][i][2]), self.image_info['labels'][i][5]-self.image_info['labels'][i][3], self.image_info['labels'][i][4]-self.image_info['labels'][i][2], edgecolor = color, fill=False)
            rect.set_linestyle('dashed')
            
            ax.add_patch(rect)
            ax.text(int(self.image_info['labels'][i][3]), int(self.image_info['labels'][i][2]) + 8, str(self.image_info['labels'][i][1]), color='w', size=11, backgroundcolor="none")
            
            rect = patches.Rectangle((self.image_info['labels'][i][3], self.image_info['labels'][i][2]), self.image_info['labels'][i][5]-self.image_info['labels'][i][3], self.image_info['labels'][i][4]-self.image_info['labels'][i][2], edgecolor = color, fill=False)
            rect.set_linestyle('dashed')
            ax.add_patch(rect)
                
            polygon = patches.Polygon(segmentation, edgecolor = color, fill=False)
            ax.add_patch(polygon)
        
            self.obj["region_{}".format(i)].append({
                "category": colors[color_index],
                "name": "polygon",
                "all_points_x": np.asarray(all_xpoints).tolist(),
                "all_points_y": np.asarray(all_ypoints).tolist(),
                "segmentation": np.asarray(segmentation_sort_into_file).tolist(),
                "bbox": np.asarray(bbox).tolist(),
                "region_attributes":{}
            })
        
        self.annotation_index = {}
        self.annotation_index["fileref"] = []
        self.annotation_index["size"] = 512 * 384
        self.annotation_index["filename"] = 'train/all_crop_v2/' + self.image_info['name'].split('\\')[-1]
        if len(color_index_total) > 1: 
            self.annotation_index["category_id"] = colors[np.max(color_index_total)]
        else:
            self.annotation_index["category_id"] = 'G'
        self.annotation_index["base64_img_data"] = []
        self.annotation_index["file_attributes"] = []
        self.annotation_index["regions"] = []
        self.annotation_index["regions"].append(self.obj)
        ax.set_axis_off()
        plt.savefig("temp/" + self.image_info["name"].split('\\')[-1], bbox_inches='tight', pad_inches=0)
        
        self.captured_modified = cv.imread("temp/" + self.image_info["name"].split('\\')[-1])
        self.captured_modified = cv.cvtColor(self.captured_modified, cv.COLOR_BGR2RGB)
        #self.captured_modified = cv.resize(self.captured_modified, (512, 384), interpolation=cv.INTER_AREA)

        rows, cols, channels = self.captured_modified.shape
        bytesPerLine = channels * cols
        QImg = QImage(self.captured_modified.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def onClicked2(self, item2):
        
        self.index_i = int(item2.text()[0])
        
        pic = cv.cvtColor(cv.imread(self.files_root + self.image_info["name"].split('\\')[-1]), cv.COLOR_BGR2RGB)
        pic = cv.resize(pic, (512, 384), interpolation=cv.INTER_AREA)
    
        fig,ax = plt.subplots(1)
        
        ax.imshow(pic)
        
        for i in range(len(self.image_info['labels'])):
            segmentation = []
            all_xpoints = []
            all_ypoints = []
            
            for x_point_id, y_point_id in zip(self.image_info['xpoints'][i], self.image_info['ypoints'][i]):
                all_xpoints.append(x_point_id) 
                all_ypoints.append(y_point_id)
                segmentation.append((x_point_id,y_point_id))
                
            bbox = [int(self.image_info['labels'][i][3]), int(self.image_info['labels'][i][2]), int(self.image_info['labels'][i][5]-int(self.image_info['labels'][i][3])), int(self.image_info['labels'][i][4])-int(self.image_info['labels'][i][2])]
            
            if str(self.image_info['labels'][i][1]) == 'R':
                color = 'red'
            elif str(self.image_info['labels'][i][1]) == 'Y':
                color = 'yellow'
            elif str(self.image_info['labels'][i][1]) == 'G':
                color = 'green'
            
            rect = patches.Rectangle((self.image_info['labels'][i][3], self.image_info['labels'][i][2]), self.image_info['labels'][i][5]-self.image_info['labels'][i][3], self.image_info['labels'][i][4]-self.image_info['labels'][i][2], edgecolor = color, fill=False)
            rect.set_linestyle('dashed')
            
            ax.add_patch(rect)
            ax.text(int(self.image_info['labels'][i][3]), int(self.image_info['labels'][i][2]) + 8, str(self.image_info['labels'][i][1]), color='w', size=11, backgroundcolor="none")
            
            if i == self.index_i-1:
                rect = patches.Rectangle((self.image_info['labels'][i][3], self.image_info['labels'][i][2]), self.image_info['labels'][i][5]-self.image_info['labels'][i][3], self.image_info['labels'][i][4]-self.image_info['labels'][i][2], edgecolor = 'blue', fill=False)
                rect.set_linestyle('dashed')
                ax.add_patch(rect)
                
                polygon = patches.Polygon(segmentation, edgecolor = 'blue', fill=False)
                ax.add_patch(polygon)
            
            else:
                rect = patches.Rectangle((self.image_info['labels'][i][3], self.image_info['labels'][i][2]), self.image_info['labels'][i][5]-self.image_info['labels'][i][3], self.image_info['labels'][i][4]-self.image_info['labels'][i][2], edgecolor = color, fill=False)
                rect.set_linestyle('dashed')
                ax.add_patch(rect)
                
                polygon = patches.Polygon(segmentation, edgecolor = color, fill=False)
                ax.add_patch(polygon)
        
        if not os.path.exists('temp/'):
            os.mkdir('temp/')
            
        ax.set_axis_off()
        plt.savefig("temp/" + self.image_info["name"].split('\\')[-1], bbox_inches='tight', pad_inches=0)
        
        self.captured_modified = cv.imread("temp/" + self.image_info["name"].split('\\')[-1])
        self.captured_modified = cv.cvtColor(self.captured_modified, cv.COLOR_BGR2RGB)
        #self.captured_modified = cv.resize(self.captured_modified, (512, 384), interpolation=cv.INTER_AREA)

        rows, cols, channels = self.captured_modified.shape
        bytesPerLine = channels * cols
        QImg = QImage(self.captured_modified.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnSave_Clicked(self):
        if self.annotation_index == {}:
            obj = {}
            obj["region_1"] = []
            obj["region_1"].append({
                "category": 'G',
                "name": "polygon",
                "all_points_x": [],
                "all_points_y": [],
                "segmentation": [],
                "bbox": [],
                "region_attributes":{}
                })
            
            self.annotation_index = {}
            self.annotation_index["fileref"] = []
            self.annotation_index["size"] = 199608
            self.annotation_index["filename"] = 'train/all_crop_v2/' + self.image_info["name"].split('\\')[-1]
            self.annotation_index["category_id"] = 'G'
            self.annotation_index["base64_img_data"] = []
            self.annotation_index["file_attributes"] = []
            self.annotation_index["regions"] = []
            self.annotation_index["regions"].append(obj)
        
        if self.index_i == -1:
            QMessageBox.information(self, 'Title','Please select the row displayed in labelbox.')
            return
            
        if not hasattr(self, "captured"):
            self.textBrowser.setPlainText("No image input")
            return
        
        pic = cv.cvtColor(cv.imread(self.files_root + self.image_info["name"].split('\\')[-1]), cv.COLOR_BGR2RGB)
        pic = cv.resize(pic, (512, 384), interpolation=cv.INTER_AREA)
    
        fig,ax = plt.subplots(1)
        
        ax.imshow(pic)
        
        for i in range(len(self.image_info['labels'])):
            segmentation = []
            all_xpoints = []
            all_ypoints = []
            
            for x_point_id, y_point_id in zip(self.image_info['xpoints'][i], self.image_info['ypoints'][i]):
                all_xpoints.append(x_point_id) 
                all_ypoints.append(y_point_id)
                segmentation.append((x_point_id,y_point_id))
                
            bbox = [int(self.image_info['labels'][i][3]), int(self.image_info['labels'][i][2]), int(self.image_info['labels'][i][5]-int(self.image_info['labels'][i][3])), int(self.image_info['labels'][i][4])-int(self.image_info['labels'][i][2])]
            
            if str(self.image_info['labels'][i][1]) == 'R':
                color = 'red'
            elif str(self.image_info['labels'][i][1]) == 'Y':
                color = 'yellow'
            elif str(self.image_info['labels'][i][1]) == 'G':
                color = 'green'
            
            rect = patches.Rectangle((self.image_info['labels'][i][3], self.image_info['labels'][i][2]), self.image_info['labels'][i][5]-self.image_info['labels'][i][3], self.image_info['labels'][i][4]-self.image_info['labels'][i][2], edgecolor = color, fill=False)
            rect.set_linestyle('dashed')
            
            ax.add_patch(rect)
            ax.text(int(self.image_info['labels'][i][3]), int(self.image_info['labels'][i][2]) + 8, str(self.image_info['labels'][i][1]), color='w', size=11, backgroundcolor="none")
            
            rect = patches.Rectangle((self.image_info['labels'][i][3], self.image_info['labels'][i][2]), self.image_info['labels'][i][5]-self.image_info['labels'][i][3], self.image_info['labels'][i][4]-self.image_info['labels'][i][2], edgecolor = color, fill=False)
            rect.set_linestyle('dashed')
            ax.add_patch(rect)
                
            polygon = patches.Polygon(segmentation, edgecolor = color, fill=False)
            ax.add_patch(polygon)
        
        ax.text(16, 16, self.image_info["name"].split('\\')[-1], color='w', size=12, backgroundcolor="none")
        ax.set_axis_off()
        plt.savefig("temp/" + self.image_info["name"].split('\\')[-1], bbox_inches='tight', pad_inches=0)
        
        self.captured_modified = cv.imread("temp/" + self.image_info["name"].split('\\')[-1])
        self.captured_modified = cv.cvtColor(self.captured_modified, cv.COLOR_BGR2RGB)
        #self.captured_modified = cv.resize(self.captured_modified, (512, 384), interpolation=cv.INTER_AREA)

        rows, cols, channels = self.captured_modified.shape
        bytesPerLine = channels * cols
        QImg = QImage(self.captured_modified.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        
        self.labelresult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelresult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        img = cv.imread("temp/" + self.image_info["name"].split('\\')[-1])
        
        if not os.path.exists('new_images/'):
            os.mkdir('new_images/')
            
        fd, type = QFileDialog.getSaveFileName(self, "Save image", 'new_images/' + self.image_info["name"].split('\\')[-1], 'png(*.png)')
        cv.imwrite(fd, img)
        
        json_files_dir = 'json_files'
        
        if not os.path.exists(json_files_dir + '/'):
            os.mkdir(json_files_dir + '/')
            
        with open(json_files_dir + '/lesion_oralCa_20200319~20200924_{}_merge.json'.format(self.image_info["name"].split('\\')[-1].split('_')[0]), 'w') as json_file:
            json.dump(self.annotation_index, json_file)
              
        self.textBrowser.append("Image saved.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    unicorn.run(app, port=8000, hast="0.0.0.0")
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())