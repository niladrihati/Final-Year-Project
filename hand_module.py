import cv2
import mediapipe as mp
import time


class HandDetector():
    def start(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Setting the mediapipe
        self.mpHands = mp.solutions.hands  # Getting the 'hands' solution to detect hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        ) # Making the object of 'Hands' class to use methods/functions
        self.mpDraw = mp.solutions.drawing_utils # Getting the 'drawing' solution/methods

        
        self.currentTime = 0
        self.previousTime = 0


    def findhands(self, result, frame):
        if result.multi_hand_landmarks:
            for Handlandmarks in result.multi_hand_landmarks:
                for index, lm in enumerate(Handlandmarks.landmark):
                    # print(index, lm) # Printing the hand landmarks (x, y, z)
                    height, width, channel = frame.shape
                    cx, cy = int(lm.x * width), int(lm.y * height) # Converting the ratios
                    # print(index, f'{cx}X', f'{cy}Y') # Printing landmarks in proper X & Y Format
                    if index == 0:
                        cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1) # Making a big palm circle
                    
                self.mpDraw.draw_landmarks(frame, Handlandmarks,
                                    self.mpHands.HAND_CONNECTIONS,
                                    self.mpDraw.DrawingSpec(color = (255, 255, 255), thickness = 2, circle_radius = 3),
                                    self.mpDraw.DrawingSpec(color = (0, 0, 255), thickness = 2))    
    
    

    def main(self):
        cap = cv2.VideoCapture(0)
        self.start()
        
        while True:
            success, frame = cap.read()
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converting the BGR to RGB
            result = self.hands.process(frameRGB) # Getting the landmarks of each frame
            self.findhands(result, frame)
            
            # Calculating frames
            self.currentTime = time.time()
            fps = 1/(self.currentTime - self.previousTime)
            self.previousTime = self.currentTime
        
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow("Webcam capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    detector = HandDetector()
    detector.main()