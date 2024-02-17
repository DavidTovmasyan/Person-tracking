import cv2
import time
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep


classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(100, 100)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Servo initialization

pi_gpio_factory = PiGPIOFactory()
servo = AngularServo(17, min_pulse_width = 0.0006, max_pulse_width = 0.0023,pin_factory = pi_gpio_factory)

# PID controller
kp = 0.029
ki = 0.01
kd = 0.001
prev_error = 0
integral = 0

def pid_control(current_value,target_value):
    global prev_error, integral
    
    
    error = target_value - current_value
    print("ERROR",error)
    integral += error
    derivative = error - prev_error
    
    output = kp*error + ki * integral + kd * derivative
    
    prev_error = error
    
    return output


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className, confidence])
                #Drawing borders 
                if draw:
                    x, y, w, h = box
                    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (x + 10, y + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                  
                
    return img, objectInfo

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 640)

    while True:
        start_time = time.time()
        success, img = cap.read()
        result, objectInfo = getObjects(img, 0.45, 0.2, objects="person")
        for info in objectInfo:
            box, className, confidence = info
            x, y, w, h = box
            x_center = box[0] + box[2] // 2
            target_x = 320
            servo_angle = pid_control(x_center, target_x)
            if servo_angle<89 and servo_angle>-89:
                servo.angle = servo_angle
            else:
                pass
        cv2.imshow("Output", img)
        cv2.waitKey(1)
        end_time = time.time()
        frame_delay = end_time - start_time
        frame_rate = 1 / frame_delay
        print(f'Frame_delay: {frame_delay: .2f} s | Frame rate : {frame_rate: .1f} FPS | Servo angle: {int(servo.angle)} degree.')
