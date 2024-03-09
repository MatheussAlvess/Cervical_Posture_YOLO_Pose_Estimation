from ultralytics import YOLO
import numpy as np
import cv2
import math

''' aux_dict_body_keys ={
        'NOSE':0,           
        'LEFT_nose':1,       
        'RIGHT_nose':2,      
        'LEFT_EAR':3,       
        'RIGHT_EAR':4,      
        'LEFT_SHOULDER':5,  
        'RIGHT_SHOULDER':6, 
        'LEFT_ELBOW':7,     
        'RIGHT_ELBOW':8,    
        'LEFT_WRIST':9,     
        'RIGHT_WRIST':10,    
        'LEFT_HIP':11,       
        'RIGHT_HIP':12,      
        'LEFT_KNEE':13,      
        'RIGHT_KNEE':14,     
        'LEFT_ANKLE':15,     
        'RIGHT_ANKLE':16} '''

class CervicalPosture:

    def __init__(self,
                 video_path: str = 'demo.mp4',
                 model_path: str = 'yolov8m-pose.pt',
                 output_name: str = 'output_video.mp4',
                 side: str = 'right',
                 conf: float = 0.3) -> None:

        self.video_path = video_path
        self.model_path = model_path
        self.output_name = output_name
        self.side = side
        self.conf = conf

        self.model = YOLO(self.model_path)

        self.neutral_color = (255, 255, 255)
        self.good_posture_color = (0, 255, 0)
        self.reasonable_posture_color = (255,0,0)
        self.bad_posture_color = (0, 0, 255)
        
    def angle_finder(self,x1, y1, x2, y2):
        theta = math.acos((y2 - y1) * (-y1) / (math.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        degree = int(180 / math.pi) * theta
        return degree

    def angle_degrees(self,A, B, C):
        # Angle of AB and AC
        AB = (B[0] - A[0], B[1] - A[1])
        AC = (C[0] - A[0], C[1] - A[1])
        # Calculate the dot product of AB and AC
        produto_escalar = AB[0] * AC[0] + AB[1] * AC[1]
        # Calculate vector magnitudes
        magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
        magnitude_AC = math.sqrt(AC[0]**2 + AC[1]**2)
        # Calculate the cosine of the angle between vectors
        cos_theta = produto_escalar / (magnitude_AB * magnitude_AC)
        # Calculate angle in radians
        theta_radians = math.acos(cos_theta)
        # Convert angle to degrees
        theta_graus = math.degrees(theta_radians)

        return theta_graus

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Get video properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
        self.out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

        while self.cap.isOpened():
            success, image = self.cap.read()
            
            if success:
                h, w, _ = image.shape
                results = self.model(image,conf=self.conf)

                keypoints = np.array(results[0].keypoints.xyn[0])

                if self.side == 'left':
                    ear = 3
                    shoulder = 5
                else:
                    ear = 4
                    shoulder = 6
                
                nose_x = int(keypoints[0][0]*w)  # right nose, x
                nose_y = int(keypoints[0][1]*h)  # right nose, y
                ear_x = int(keypoints[ear][0]*w)  # right ear, x
                ear_y = int(keypoints[ear][1]*h)  # right ear, y
                shoulder_x = int(keypoints[shoulder][0]*w)  # right shoulder, x
                shoulder_y = int(keypoints[shoulder][1]*h)  # right shoulder, y

                cv2.line(image, (shoulder_x, shoulder_y), (shoulder_x, shoulder_y-200), self.neutral_color, 2) #shoulder to aux conection
                cv2.line(image, (shoulder_x, shoulder_y), (ear_x, ear_y), self.neutral_color, 2) #shoulder to ear conection
                cv2.line(image, (ear_x, ear_y), (nose_x, nose_y), self.neutral_color, 2) #ear to nose conection

                cv2.circle(image, (nose_x, nose_y), 8, self.neutral_color, -1) #nose point
                cv2.circle(image, (ear_x, ear_y), 8, self.neutral_color, -1) #ear point
                cv2.circle(image, (shoulder_x, shoulder_y), 8, self.neutral_color, -1) #shoulder point
                cv2.circle(image, (shoulder_x, shoulder_y-200), 8, self.neutral_color, -1) #aux point

                neck_inclination = self.angle_finder(shoulder_x,shoulder_y,ear_x,ear_y) #shoulder to ear inclination
                CRA_inclination = self.angle_degrees((ear_x,ear_y),(shoulder_x,shoulder_y),(nose_x,nose_y)) #shoulder to ear inclination

                angle_text_string = 'Degrees -> Neck : ' + str(int(neck_inclination)) + ' CRA : ' + str(int(CRA_inclination)) 

                def posture(posture_color,text):
                    cv2.putText(image, angle_text_string, (10, 30), self.font, 0.9, posture_color, 2)
                    cv2.putText(image, text, (10, 60), self.font, 0.9, posture_color, 2)
                    cv2.putText(image, str(int(neck_inclination)), (shoulder_x + 10, shoulder_y), self.font, 0.9, posture_color, 2)
                    cv2.putText(image, str(int(CRA_inclination)), (ear_x + 10, ear_y), self.font, 0.9, posture_color, 2)

                    cv2.line(image, (shoulder_x, shoulder_y), (ear_x, ear_y), posture_color, 3)
                    cv2.line(image, (shoulder_x, shoulder_y), (shoulder_x, shoulder_y - 200), self.neutral_color, 2)
                    cv2.line(image, (nose_x, nose_y), (ear_x, ear_y), posture_color, 3)

                if (int(neck_inclination) <= 15) and (int(CRA_inclination) >= 85) and (int(CRA_inclination) <= 95):
                    good_text = 'Good Posture'
                    posture(self.good_posture_color,good_text)

                elif (int(neck_inclination) <= 20) and (int(CRA_inclination) >= 80) and (int(CRA_inclination) <= 100):
                    reasonable_text = 'Reasonable Posture'
                    posture(self.reasonable_posture_color,reasonable_text)
                else:
                    bad_text = 'Bad Posture'
                    posture(self.bad_posture_color,bad_text)

                self.out.write(image)
                cv2.imshow('Yolo inference',image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    cervical_analysis = CervicalPosture(video_path='demo.mp4',model_path='yolov8m-pose.pt',side='left')
    cervical_analysis.run()