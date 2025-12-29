# workout_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque

class WorkoutDetector:
    def __init__(self):
        # Load YOLO pose model
        self.model = YOLO('yolov8n-pose.pt')  # Download automatically on first run
        
        # Exercise states
        self.exercise_type = "pushup"  # pushup, squat, pullup
        self.counter = 0
        self.stage = None  # up, down
        
        # Smoothing buffer for angles (increased for stability)
        self.angle_buffer = deque(maxlen=8)
        
        # Exercise parameters (adjusted for better detection)
        self.pushup_down_threshold = 110  # Angle when arm is bent (down position)
        self.pushup_up_threshold = 150    # Angle when arm is extended (up position)
        self.squat_down_threshold = 90
        self.squat_up_threshold = 160
        self.pullup_down_threshold = 160
        self.pullup_up_threshold = 90
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)  # First point
        b = np.array(b)  # Mid point
        c = np.array(c)  # End point
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) -                   np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def smooth_angle(self, angle):
        """Apply moving average smoothing"""
        self.angle_buffer.append(angle)
        return np.mean(self.angle_buffer)
    
    def detect_pushup(self, keypoints):
        """Detect push-up exercise - improved version"""
        # Get keypoint coordinates
        # 5=left_shoulder, 7=left_elbow, 9=left_wrist
        # 6=right_shoulder, 8=right_elbow, 10=right_wrist
        
        try:
            # Use average of both arms for more stability
            left_shoulder = [keypoints[5][0], keypoints[5][1]]
            left_elbow = [keypoints[7][0], keypoints[7][1]]
            left_wrist = [keypoints[9][0], keypoints[9][1]]
            
            right_shoulder = [keypoints[6][0], keypoints[6][1]]
            right_elbow = [keypoints[8][0], keypoints[8][1]]
            right_wrist = [keypoints[10][0], keypoints[10][1]]
            
            # Calculate both elbow angles
            left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Use average for more stable detection
            angle = (left_angle + right_angle) / 2
            angle = self.smooth_angle(angle)
            
            # Improved state machine with hysteresis
            if angle < self.pushup_down_threshold and self.stage != "down":
                self.stage = "down"
            if angle > self.pushup_up_threshold and self.stage == "down":
                self.stage = "up"
                self.counter += 1
                
            return angle, True
        except:
            return 0, False
    
    def detect_squat(self, keypoints):
        """Detect squat exercise"""
        # 11=left_hip, 13=left_knee, 15=left_ankle
        # 12=right_hip, 14=right_knee, 16=right_ankle
        
        try:
            # Use right leg for detection
            hip = [keypoints[12][0], keypoints[12][1]]
            knee = [keypoints[14][0], keypoints[14][1]]
            ankle = [keypoints[16][0], keypoints[16][1]]
            
            # Calculate knee angle
            angle = self.calculate_angle(hip, knee, ankle)
            angle = self.smooth_angle(angle)
            
            # State machine for counting
            if angle < self.squat_down_threshold:
                self.stage = "down"
            if angle > self.squat_up_threshold and self.stage == "down":
                self.stage = "up"
                self.counter += 1
                
            return angle, True
        except:
            return 0, False
    
    def detect_pullup(self, keypoints):
        """Detect pull-up exercise"""
        try:
            # Use right arm for detection
            shoulder = [keypoints[6][0], keypoints[6][1]]
            elbow = [keypoints[8][0], keypoints[8][1]]
            wrist = [keypoints[10][0], keypoints[10][1]]
            
            # Calculate elbow angle
            angle = self.calculate_angle(shoulder, elbow, wrist)
            angle = self.smooth_angle(angle)
            
            # State machine (inverted from push-up)
            if angle < self.pullup_up_threshold:
                self.stage = "up"
            if angle > self.pullup_down_threshold and self.stage == "up":
                self.stage = "down"
                self.counter += 1
                
            return angle, True
        except:
            return 0, False
    
    def draw_landmarks(self, frame, keypoints):
        """Draw pose landmarks on frame"""
        # Define connections between keypoints
        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start = tuple(map(int, keypoints[start_idx][:2]))
                end = tuple(map(int, keypoints[end_idx][:2]))
                cv2.line(frame, start, end, (0, 255, 0), 2)
        
        # Draw keypoints
        for point in keypoints:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    def process_frame(self, frame):
        """Process single frame"""
        # Run YOLO pose detection
        results = self.model(frame, verbose=False)
        
        # Get keypoints
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            # Draw landmarks
            self.draw_landmarks(frame, keypoints)
            
            # Detect exercise based on type
            if self.exercise_type == "pushup":
                angle, success = self.detect_pushup(keypoints)
                exercise_name = "Push-up"
            elif self.exercise_type == "squat":
                angle, success = self.detect_squat(keypoints)
                exercise_name = "Squat"
            elif self.exercise_type == "pullup":
                angle, success = self.detect_pullup(keypoints)
                exercise_name = "Pull-up"
            
            # Display information
            if success:
                # Draw angle
                cv2.putText(frame, f'Angle: {int(angle)}', 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                # Draw stage
                cv2.putText(frame, f'Stage: {self.stage}', 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
        
        # Draw counter and exercise type
        cv2.rectangle(frame, (0, 0), (400, 60), (245, 117, 16), -1)
        cv2.putText(frame, f'{exercise_name} Counter: {self.counter}', 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        return frame
    
    def select_video_source(self):
        """Let user choose between webcam or video file"""
        print("" + "="*50)
        print("WORKOUT DETECTOR - VIDEO SOURCE SELECTION")
        print("="*50)
        print("Select video source:")
        print("1. Webcam (default)")
        print("2. Video file")
        print("3. External camera (if available)")
        
        while True:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                print("Using webcam (camera 0)...")
                return 0
            
            elif choice == '2':
                video_path = input("Enter video file path: ").strip()
                # Remove quotes if user pastes path with quotes
                video_path = video_path.strip('"').strip("'")
                
                if not video_path:
                    print("Error: No path provided. Using webcam instead.")
                    return 0
                
                # Check if file exists
                import os
                if not os.path.exists(video_path):
                    print(f"Error: File not found: {video_path}")
                    retry = input("Try again? (y/n): ").strip().lower()
                    if retry == 'y':
                        continue
                    else:
                        print("Using webcam instead.")
                        return 0
                
                print(f"Using video file: {video_path}")
                return video_path
            
            elif choice == '3':
                camera_id = input("Enter camera ID (usually 1 or 2): ").strip()
                try:
                    camera_id = int(camera_id)
                    print(f"Using external camera {camera_id}...")
                    return camera_id
                except ValueError:
                    print("Error: Invalid camera ID. Using webcam instead.")
                    return 0
            
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    def run(self):
        """Main loop for video capture"""
        # Let user select video source
        video_source = self.select_video_source()
        cap = cv2.VideoCapture(video_source)
        
        # Check if video source opened successfully
        if not cap.isOpened():
            print("Error: Could not open video source!")
            print("Please check your camera connection or video file path.")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("" + "="*50)
        print("WORKOUT DETECTOR - READY")
        print("="*50)
        print(f"Video source: {video_source}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps if fps > 0 else 'N/A'}")
        print("Controls:")
        print("  'q' - Quit")
        print("  '1' - Switch to Push-ups")
        print("  '2' - Switch to Squats")
        print("  '3' - Switch to Pull-ups")
        print("  'r' - Reset counter")
        print("  'p' - Pause/Resume")
        print("="*50 + "")
        
        paused = False
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or camera disconnected.")
                    # If it's a video file, ask if user wants to replay
                    if isinstance(video_source, str):
                        replay = input("Replay video? (y/n): ").strip().lower()
                        if replay == 'y':
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.counter = 0
                            self.stage = None
                            continue
                    break
                
                # Process frame
                frame = self.process_frame(frame)
            
            # Add pause indicator if paused
            if paused:
                cv2.putText(frame, 'PAUSED - Press P to resume', 
                           (width//2 - 200, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 2)
            
            # Display
            cv2.imshow('Workout Detector', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('1'):
                self.exercise_type = "pushup"
                self.counter = 0
                self.stage = None
                self.angle_buffer.clear()
                print("✓ Switched to Push-ups")
            elif key == ord('2'):
                self.exercise_type = "squat"
                self.counter = 0
                self.stage = None
                self.angle_buffer.clear()
                print("✓ Switched to Squats")
            elif key == ord('3'):
                self.exercise_type = "pullup"
                self.counter = 0
                self.stage = None
                self.angle_buffer.clear()
                print("✓ Switched to Pull-ups")
            elif key == ord('r'):
                self.counter = 0
                self.stage = None
                self.angle_buffer.clear()
                print("✓ Counter reset")
            elif key == ord('p'):
                paused = not paused
                print("⏸ Paused" if paused else "▶ Resumed")
        
        cap.release()
        cv2.destroyAllWindows()
        print("" + "="*50)
        print("WORKOUT DETECTOR - SESSION ENDED")
        print("="*50)
        print(f"Final {self.exercise_type.capitalize()} count: {self.counter}")
        print("Thank you for using Workout Detector!")
        print("="*50)

if __name__ == "__main__":
    detector = WorkoutDetector()
    detector.run()