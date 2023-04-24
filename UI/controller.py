#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GUI
import sys, time, os, datetime
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.Qt import *
from controller_window import Ui_Form

# Realsense
import pyrealsense2 as rs

# ROS and create the /home/$USER/DataYO
import rospy, math
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseWithCovariance
from move_base_msgs.msg import MoveBaseActionGoal
from nav_msgs.msg import Odometry
import PIL
from PIL import Image
import numpy as np
import yaml
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# YOLOv5
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

bridge = CvBridge()

robot_pose = [0, 0, 0]
robot_orientation = [0, 0, 0, 1]
detected_objects = []

W = 640
H = 480
cv_color_image = np.zeros((H, W, 3), np.uint8)
cv_depth_image = np.zeros((H, W), np.uint8)

# save file continuous form without access
save_pathC=os.environ['HOME'] + "/DataYO"
file_nameC="DataDYC"+time.strftime("_%Y%m%d_%H%M%S")+".csv"
PathFilenameC=os.path.join(save_pathC,file_nameC)
flC=open(PathFilenameC,"w")
flC.write("date"+","+"time"+","+"x"+","+"y"+","+"distance"+","+"label"+","+"percentage")
flC.write('\n')

class GraphicsScene(QGraphicsScene):

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.opt = ""

    def setOption(self, opt):
        self.opt = opt

class GraphicsScene2(QGraphicsScene):

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, 0, 0, 640, 480, parent=None)
        self.opt = ""

class GUI(QDialog):

    first_time = True

    def __init__(self,parent=None):
        # GUI
        super(GUI, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

        self.base_map_dir = os.environ['ROS_WORKSPACES'] + "/src/mobrob_mark/maps/mob_rob_2023_01_30__15_33_22.pgm"

        yaml_path = os.environ['ROS_WORKSPACES'] + "/src/mobrob_mark/maps/mob_rob_2023_01_30__15_33_22.yaml"
        with open(yaml_path) as f:
            map_data = yaml.safe_load(f)


        self.yaml_resolution = float(map_data["resolution"])
        print(f"self.yaml_resolution : {self.yaml_resolution}")

        self.yaml_origin = map_data["origin"]
        print(f"self.yaml_origin : {self.yaml_origin}")

        map_image = PIL.Image.open(self.base_map_dir)
        self.img_data = np.array(map_image)
        self.qimg = QImage(self.img_data.data, self.img_data.shape[1], self.img_data.shape[0], QImage.Format_Indexed8) 
        self.bounds = self.find_bounds(map_image)
        print(f"bounds : {self.bounds}")

        #self.ui.camera_slider.setValue(0)
        #self.pub_camera_angle = rospy.Publisher('/camera_stay_controller/command', Float64, queue_size = 10)

        #YOLOv5
        weights='yolov5s.pt'  # model.pt path(s)
        self.imgsz=640  # inference size (pixels)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.project='runs/val'  # save to project/name
        self.name='features' # save results to project/name
        self.stride = 32
        device_num=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu

        # Initialize
        set_logging()
        self.device = select_device(device_num)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet50', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(self.device).eval()

        # Dataloader
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once     

        #self.fx = 379.4980
        #self.fy = 379.4980
        #self.cx = 321.9203 # Realsense camera width resolution is 640
        #self.cy = 240.9855 # Realsense camera heigh resolution is 480

        self.fx = 347.9976
        self.fy = 347.9976
        self.cx = 320
        self.cy = 240

        #self.move_base_msg = MoveBaseActionGoal()
        #self.pub_goal = rospy.Publisher('/control/move_base', MoveBaseActionGoal, queue_size=10)
        self.goal = PoseStamped()
        self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

    def repaint(self, img, X, Y, yaw, targets):
        global robot_pose, robot_orientation#, cv_color_image

        if(self.first_time):
            self.scene = GraphicsScene(self.ui.graphicsView_2) # map
            self.scene2 = GraphicsScene2(self.ui.graphicsView) # camera image
            self.first_time = False

        # for map
        self.pixmap = QPixmap.fromImage(self.qimg)
        self.item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.item)
        self.ui.graphicsView_2.setScene(self.scene)

        #for camera image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixmap2cv = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.pixmap_camera_img = QPixmap(pixmap2cv)
        self.item = QGraphicsPixmapItem(self.pixmap_camera_img)
        self.scene2.addItem(self.item)
        self.ui.graphicsView.setScene(self.scene2)

        C = 30
        robot_vertices = [[(X + C*(- 0.25*math.sin(yaw))),                       (Y + C*(0.25*math.cos(yaw)))], 
                          [(X + C*(-0.2*math.cos(yaw) - 0.15*math.sin(yaw))),    (Y + C*(-0.2*math.sin(yaw) + 0.15*math.cos(yaw)))], 
                          [(X + C*(-0.2*math.cos(yaw) - (-0.15)*math.sin(yaw))), (Y + C*(-0.2*math.sin(yaw) - 0.15*math.cos(yaw)))],
                          [(X + C*(0.2*math.cos(yaw) - (-0.15)*math.sin(yaw))),  (Y + C*(0.2*math.sin(yaw) - 0.15*math.cos(yaw)))], 
                          [(X + C*(0.2*math.cos(yaw) - 0.15*math.sin(yaw))),     (Y + C*(0.2*math.sin(yaw) + 0.15*math.cos(yaw)))]]

        qpoly_robot = QPolygonF([QPointF(p[0], p[1]) for p in robot_vertices])
        pen_robot = QPen(Qt.red)
        brush_robot = QBrush(Qt.red)
        pen_robot.setWidth(2)
        self.scene.addPolygon(qpoly_robot, pen_robot, brush_robot)
        for i in range(len(targets)):
            self.scene.addEllipse(targets[i][0] - 6, targets[i][1] - 6, 12, 12, QPen(Qt.green), QBrush(Qt.green))

    def find_bounds(self, map_image):
        x_min = map_image.size[0]
        x_end = 0
        y_min = map_image.size[1]
        y_end = 0
        pix = map_image.load()
        for x in range(map_image.size[0]):
            for y in range(map_image.size[1]):
                val = pix[x, y]
                if val != 205:
                    x_min = min(x, x_min)
                    x_end = max(x, x_end)
                    y_min = min(y, y_min)
                    y_end = max(y, y_end)
        return x_min, x_end, y_min, y_end

    def update(self):
        # save file per frame (2 seconds) with access
        save_pathF=os.environ['HOME'] + "/DataYO"
        file_nameF="DataDYF"+time.strftime("_%Y%m%d_%H%M%S")+".csv"
        PathFilenameF=os.path.join(save_pathF,file_nameF)
        flF=open(PathFilenameF,"w")
        flF.write("date"+","+"time"+","+"x"+","+"y"+","+"distance"+","+"label"+","+"percentage")
        flF.write('\n')

        global cv_color_image, cv_depth_image, robot_pose, robot_orientation, detected_objects

        ### control the camera angle
        #camera_angle = -math.pi * float(self.ui.camera_slider.value())/180
        #self.pub_camera_angle.publish(camera_angle)

        ### calculate the robot pose
        X_robot_in_map = (robot_pose[0] - float(self.yaml_origin[0])) / self.yaml_resolution
        Y_robot_in_map = -robot_pose[1] / self.yaml_resolution + (self.bounds[3] + float(self.yaml_origin[1]) / self.yaml_resolution)

        q0 = robot_orientation[0]
        q1 = robot_orientation[1]
        q2 = robot_orientation[2]
        q3 = robot_orientation[3]

        yaw = math.atan2(2*(q0*q1 + q2*q3),(q0**2 - q1**2 - q2**2 + q3**2))
        yaw = -yaw - math.pi/2

        ### YOLO
        # check for common shapes
        s = np.stack([letterbox(x, self.imgsz, stride=self.stride)[0].shape for x in cv_color_image], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        # Letterbox
        img0 = cv_color_image.copy()
        img = cv_color_image[np.newaxis, :, :, :]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)        

        # Inference
        t1 = time_synchronized()
        pred = self.model(img,
                     augment=self.augment,
                     visualize=increment_path(Path(self.project) / self.name, mkdir=True) if self.visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, img0)

        targets = [] # detected objects
        detected_objects.clear()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]},{conf:.2f}')
                    #print(f"(xyxy) : {xyxy}")
                    #print(f"(xyxy[0],xyxy[1],xyxy[2],xyxy[3]) : {xyxy[0]},{xyxy[1]},{xyxy[2]},{xyxy[3]},")
                    x_center = int((xyxy[0] + xyxy[2])/2)
                    y_center = int((xyxy[1] + xyxy[3])/2)
                    #print(f"(x_center, y_center) : {x_center}, {y_center}")
                    dist = float(cv_depth_image[y_center, x_center])/1000
                    #print(f"(FLOATCV, dist) : {float(cv_depth_image[y_center, x_center])}, {dist}")
                    X = dist*(x_center - self.cx)/self.fx
                    Y = dist*(y_center - self.cy)/self.fy
                    Z = dist
                    #text = "X : " + str(round(X,3)) + ", Y : " + str(round(Y, 3)) + ", Z : " + str(round(Z, 3))
                    #plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=self.line_thickness)    
                    #cv2.putText(img0, text, (int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2)), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
                    #print(f"(text, label, X, Y) : {text}, {label}, {int((xyxy[0] + xyxy[2])/2)}, {int((xyxy[1] + xyxy[3])/2)}")

                    camera_angle=-0.0

                    # convert to robot coordinates
                    Y_robot = -(X*math.cos(camera_angle) - Z*math.sin(camera_angle))
                    X_robot = X*math.sin(camera_angle) + Z*math.cos(camera_angle)
                    #print(f"(X_robot, Y_robot) : {X_robot}, {Y_robot}")

                    # convert to map coordinates
                    yaw_robot = -math.pi/2 - yaw
                    print(f"(yaw, yaw_robot) : {yaw}, {yaw_robot}")

                    X_map = X_robot*math.cos(yaw_robot) - Y_robot*math.sin(yaw_robot) + robot_pose[0]
                    Y_map = X_robot*math.sin(yaw_robot) + Y_robot*math.cos(yaw_robot) + robot_pose[1]
                    #print(f"(X_map, Y_map) : {X_map}, {Y_map}")
                    
                    text = "X:" + str(round(X_map,3)) + ", Y:" + str(round(Y_map, 3)) + ", Dis:" + str(round(Z, 3))
                    textfl =str(round(X_map,3))+","+ str(round(Y_map, 3))+","+str(round(Z, 3))
                    plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=self.line_thickness)    
                    cv2.putText(img0, text, (int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2)), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
                    print(f"(text, label, X_map, Y_map) : {text}, {label}, {int((xyxy[0] + xyxy[2])/2)}, {int((xyxy[1] + xyxy[3])/2)}")
                    flC.write(time.strftime("%Y%m%d,%H%M%S")+","+textfl+","+label)
                    flC.write('\n')
                    flF.write(time.strftime("%Y%m%d,%H%M%S")+","+textfl+","+label)
                    flF.write('\n')                    

                    detected_objects.append([X_map, Y_map])
                    # convert to pixel
                    X_target = (X_map - float(self.yaml_origin[0])) / self.yaml_resolution
                    Y_target = -Y_map / self.yaml_resolution + (self.bounds[3] + float(self.yaml_origin[1]) / self.yaml_resolution)
                    targets.append([X_target, Y_target]) # to plot detected objects on the map
        # fl.close()
        ### YOLO
        if(len(targets) > 0):
            print(f"targets : {targets}")
        self.repaint(img0, X_robot_in_map, Y_robot_in_map, yaw, targets)


    def pixmap_to_cv(self, pixmap):
        qimage = pixmap.toImage()
        w, h, d = qimage.size().width(), qimage.size().height(), qimage.depth()
        bytes_ = qimage.bits().asstring(w * h * d // 8)
        arr = np.frombuffer(bytes_, dtype=np.uint8).reshape((h, w, d // 8))
        im_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return im_bgr
            
    def measure(self):
        if(self.start_measurment == False):
            self.start_measurment = True
            self.time_start = time.time()

    def navigate(self):
        global robot_pose, robot_orientation, detected_objects
        min_dist = 100
        target_coordinates = [0, 0]
        goal_coordinates = [0, 0]
        a = 0.5 # meters
        set_goal = False
        if(len(detected_objects) > 0):
            for i in range(len(detected_objects)):
                dist = math.sqrt((robot_pose[0] - detected_objects[i][0])**2 + (robot_pose[1] - detected_objects[i][1])**2)
                if(dist < min_dist):
                    min_dist = dist
                    target_coordinates[0] = detected_objects[i][0]
                    target_coordinates[1] = detected_objects[i][1]      
                    set_goal = True
            if(set_goal):
                if(robot_pose[0] < target_coordinates[0]):
                    goal_coordinates[0] = target_coordinates[0] - a
                else:
                    goal_coordinates[0] = target_coordinates[0] + a
                
                print(f"(robot_pose[0],robot_pose[1],target_coordinates[0],target_coordinates[1],goal_coordinates[0]) : {robot_pose[0]}, {robot_pose[1]},{target_coordinates[1]},{target_coordinates[1]},{goal_coordinates[0]}")
                goal_coordinates[1] = ((robot_pose[1] - target_coordinates[1])/(robot_pose[0] - target_coordinates[0]))*(goal_coordinates[0]- robot_pose[0]) + robot_pose[1]
        else:
            print("No detected objects!")
            set_goal = False

        if(set_goal):

            self.goal.header.stamp = rospy.Time.now()
            self.goal.header.frame_id = "map"
            self.goal.pose.position.x = goal_coordinates[0]
            self.goal.pose.position.y = goal_coordinates[1]
            self.goal.pose.position.z = 0
            self.goal.pose.orientation.z = 0
            self.goal.pose.orientation.w = 1

            self.pub_goal.publish(self.goal)
        else:
            print("There is no goal!")
            
def get_robot_pose(data):
    global robot_pose, robot_orientation
    robot_pose[0] = data.pose.pose.position.x
    robot_pose[1] = data.pose.pose.position.y
    robot_orientation[0] = data.pose.pose.orientation.x
    robot_orientation[1] = data.pose.pose.orientation.y
    robot_orientation[2] = data.pose.pose.orientation.z
    robot_orientation[3] = data.pose.pose.orientation.w
    #print(f"(robot_pose[0], robot_pose[1]) : {robot_pose[0]}, {robot_pose[1]}")

def callback_color_img(data):
    global cv_color_image
    cv_color_image = bridge.imgmsg_to_cv2(data, "bgr8")

def callback_depth_img(data):
    global cv_depth_image
    cv_depth_image = bridge.imgmsg_to_cv2(data, "16UC1")

rospy.init_node('controller_nn')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUI()
    rospy.Subscriber("/odom", Odometry, get_robot_pose)
    # rospy.Subscriber("/camera/color/image_raw", Image, callback_color_img)
    # rospy.Subscriber("/camera/depth/image_rect_raw", Image, callback_depth_img)
    # rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, get_robot_pose)
    rospy.Subscriber("/zed_node/rgb/image_rect_color", Image, callback_color_img)
    rospy.Subscriber("/zed_node/depth/depth_registered", Image, callback_depth_img)
    window.show()
    sys.exit(app.exec_())
