import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def detect_hand(self, image):
        """
        Detect hand landmarks in the image
        Returns: landmarks (list of coordinates), annotated_image
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        # Make a copy for annotation
        annotated_image = image.copy()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
            
            # Draw hand landmarks with custom styling
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(255, 255, 255), thickness=2)
            )
            
            # Convert landmarks to pixel coordinates
            h, w, _ = image.shape
            landmarks = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            # Draw palm lines for visualization
            annotated_image = self._draw_palm_lines(annotated_image, landmarks)
            
            return landmarks, annotated_image
        
        return None, annotated_image
    
    def _draw_palm_lines(self, image, landmarks):
        """
        Draw estimated palm lines on the image for visualization
        """
        if len(landmarks) < 21:
            return image
        
        # Define colors for different lines
        life_color = (0, 255, 0)    # Green
        heart_color = (0, 0, 255)   # Red  
        head_color = (255, 0, 0)    # Blue
        fate_color = (0, 255, 255)  # Yellow
        
        # Life line (curves around thumb)
        life_start = ((landmarks[1][0] + landmarks[5][0]) // 2, 
                     (landmarks[1][1] + landmarks[5][1]) // 2)
        life_end = landmarks[0]
        cv2.line(image, life_start, life_end, life_color, 3)
        
        # Heart line (horizontal below fingers)
        heart_start = (landmarks[17][0], landmarks[17][1] + 20)
        heart_end = (landmarks[5][0], landmarks[5][1] + 20)
        cv2.line(image, heart_start, heart_end, heart_color, 3)
        
        # Head line (horizontal in middle of palm)
        head_start = ((landmarks[1][0] + landmarks[5][0]) // 2 + 10, 
                     (landmarks[1][1] + landmarks[5][1]) // 2 + 30)
        head_end = (landmarks[17][0], head_start[1])
        cv2.line(image, head_start, head_end, head_color, 3)
        
        # Fate line (vertical up the palm)
        fate_center_x = (landmarks[5][0] + landmarks[17][0]) // 2
        fate_start = (fate_center_x, landmarks[0][1])
        fate_end = (fate_center_x, landmarks[9][1])
        cv2.line(image, fate_start, fate_end, fate_color, 3)
        
        # Add line labels
        cv2.putText(image, 'Life', (life_start[0] - 20, life_start[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, life_color, 2)
        cv2.putText(image, 'Heart', (heart_start[0], heart_start[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, heart_color, 2)
        cv2.putText(image, 'Head', (head_start[0], head_start[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_color, 2)
        cv2.putText(image, 'Fate', (fate_center_x + 5, fate_end[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fate_color, 2)
        
        return image
    
    def get_hand_shape_type(self, landmarks):
        """
        Determine hand shape type based on finger proportions
        Returns: shape type (square, pointed, spatulate, conic)
        """
        if not landmarks or len(landmarks) < 21:
            return "unknown"
        
        try:
            # Calculate finger lengths and palm dimensions
            wrist = landmarks[0]
            middle_tip = landmarks[12]
            middle_mcp = landmarks[9]  # Middle finger base
            index_mcp = landmarks[5]   # Index finger base
            pinky_mcp = landmarks[17]  # Pinky finger base
            
            # Palm width (approximate between index and pinky base)
            palm_width = abs(index_mcp[0] - pinky_mcp[0])
            
            # Palm length (wrist to middle finger base)
            palm_length = abs(middle_mcp[1] - wrist[1])
            
            # Finger length (middle finger from base to tip)
            finger_length = abs(middle_tip[1] - middle_mcp[1])
            
            # Avoid division by zero
            if palm_width == 0 or palm_length == 0:
                return "unknown"
            
            palm_ratio = palm_length / palm_width
            finger_palm_ratio = finger_length / palm_length if palm_length > 0 else 1
            
            # Classify hand shape based on ratios
            if palm_ratio < 0.85:  # Wide palm
                if finger_palm_ratio < 0.75:
                    return "square"
                else:
                    return "spatulate"
            elif palm_ratio > 1.15:  # Long palm
                if finger_palm_ratio > 0.9:
                    return "pointed"
                else:
                    return "conic"
            else:  # Balanced palm
                if finger_palm_ratio < 0.7:
                    return "spatulate"
                elif finger_palm_ratio > 0.95:
                    return "pointed"
                else:
                    return "conic"
        
        except Exception as e:
            print(f"Error calculating hand shape: {e}")
            return "unknown"
    
    def get_palm_center(self, landmarks):
        """
        Calculate the center point of the palm
        """
        if not landmarks or len(landmarks) < 21:
            return None
        
        try:
            # Use key palm points to calculate center
            palm_points = [
                landmarks[0],   # Wrist
                landmarks[1],   # Thumb base
                landmarks[5],   # Index finger base
                landmarks[9],   # Middle finger base
                landmarks[13],  # Ring finger base
                landmarks[17]   # Pinky base
            ]
            
            center_x = sum(point[0] for point in palm_points) // len(palm_points)
            center_y = sum(point[1] for point in palm_points) // len(palm_points)
            
            return (center_x, center_y)
        
        except Exception as e:
            print(f"Error calculating palm center: {e}")
            return None
    
    def get_finger_lengths(self, landmarks):
        """
        Calculate relative finger lengths
        Returns: dict with finger length ratios
        """
        if not landmarks or len(landmarks) < 21:
            return None
        
        try:
            # Finger tip and base coordinates
            fingers = {
                'thumb': {'tip': landmarks[4], 'base': landmarks[1]},
                'index': {'tip': landmarks[8], 'base': landmarks[5]},
                'middle': {'tip': landmarks[12], 'base': landmarks[9]},
                'ring': {'tip': landmarks[16], 'base': landmarks[13]},
                'pinky': {'tip': landmarks[20], 'base': landmarks[17]}
            }
            
            finger_lengths = {}
            for finger_name, coords in fingers.items():
                length = np.sqrt(
                    (coords['tip'][0] - coords['base'][0])**2 + 
                    (coords['tip'][1] - coords['base'][1])**2
                )
                finger_lengths[finger_name] = length
            
            # Calculate relative lengths (compared to middle finger)
            middle_length = finger_lengths['middle']
            if middle_length > 0:
                for finger in finger_lengths:
                    finger_lengths[finger] = finger_lengths[finger] / middle_length
            
            return finger_lengths
        
        except Exception as e:
            print(f"Error calculating finger lengths: {e}")
            return None
    
    def analyze_hand_flexibility(self, landmarks):
        """
        Analyze hand flexibility based on finger curvature
        Returns: flexibility score (0-1)
        """
        if not landmarks or len(landmarks) < 21:
            return 0.5  # Default moderate flexibility
        
        try:
            # Calculate angles between finger joints
            flexibility_score = 0
            finger_count = 0
            
            # Analyze each finger's curvature
            fingers = [
                [landmarks[2], landmarks[3], landmarks[4]],    # Thumb
                [landmarks[6], landmarks[7], landmarks[8]],    # Index
                [landmarks[10], landmarks[11], landmarks[12]], # Middle
                [landmarks[14], landmarks[15], landmarks[16]], # Ring
                [landmarks[18], landmarks[19], landmarks[20]]  # Pinky
            ]
            
            for finger_joints in fingers:
                if len(finger_joints) >= 3:
                    # Calculate angle at middle joint
                    p1, p2, p3 = finger_joints
                    
                    # Vector from p2 to p1 and p2 to p3
                    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                    
                    # Calculate angle
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range
                    angle = np.arccos(cos_angle)
                    
                    # Convert to flexibility score (more curved = more flexible)
                    flexibility_score += 1 - (angle / np.pi)
                    finger_count += 1
            
            return flexibility_score / finger_count if finger_count > 0 else 0.5
        
        except Exception as e:
            print(f"Error analyzing hand flexibility: {e}")
            return 0.5
    
    def get_hand_quality_metrics(self, landmarks, image):
        """
        Calculate quality metrics for the hand detection
        Returns: dict with quality scores
        """
        if not landmarks or len(landmarks) < 21:
            return {'overall': 0, 'visibility': 0, 'clarity': 0}
        
        try:
            # Calculate visibility score based on landmark confidence
            visibility_score = 1.0  # MediaPipe doesn't provide confidence in this version
            
            # Calculate clarity based on hand area and resolution
            hand_area = self._calculate_hand_area(landmarks)
            image_area = image.shape[0] * image.shape[1]
            hand_ratio = hand_area / image_area
            
            clarity_score = min(hand_ratio * 10, 1.0)  # Scale to 0-1
            
            # Overall quality
            overall_score = (visibility_score + clarity_score) / 2
            
            return {
                'overall': overall_score,
                'visibility': visibility_score,
                'clarity': clarity_score,
                'hand_area_ratio': hand_ratio
            }
        
        except Exception as e:
            print(f"Error calculating quality metrics: {e}")
            return {'overall': 0.5, 'visibility': 0.5, 'clarity': 0.5}
    
    def _calculate_hand_area(self, landmarks):
        """
        Calculate approximate hand area using convex hull
        """
        try:
            # Convert landmarks to numpy array
            points = np.array(landmarks, dtype=np.int32)
            
            # Calculate convex hull
            hull = cv2.convexHull(points)
            
            # Calculate area
            area = cv2.contourArea(hull)
            
            return area
        
        except Exception as e:
            print(f"Error calculating hand area: {e}")
            return 0
    
    def cleanup(self):
        """
        Clean up resources
        """
        if hasattr(self, 'hands'):
            self.hands.close()