import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import splrep, splev
import math

class LineExtractor:
    def __init__(self):
        self.line_colors = {
            'life': (0, 255, 0),    # Green
            'heart': (0, 0, 255),   # Red
            'head': (255, 0, 0),    # Blue
            'fate': (255, 255, 0)   # Yellow
        }
        self.debug_mode = False
    
    def extract_lines(self, image, landmarks):
        """
        Extract palm lines from the hand image using landmarks
        Returns: dictionary with line information
        """
        try:
            lines_info = {
                'life': None,
                'heart': None,
                'head': None,
                'fate': None
            }
            
            if not landmarks or len(landmarks) < 21:
                return lines_info
            
            # Create palm region mask
            palm_mask = self._create_palm_mask(image, landmarks)
            
            # Preprocess image for line detection
            processed_image = self._preprocess_for_lines(image, palm_mask)
            
            # Extract each major line
            lines_info['life'] = self._extract_life_line(processed_image, landmarks, image)
            lines_info['heart'] = self._extract_heart_line(processed_image, landmarks, image)
            lines_info['head'] = self._extract_head_line(processed_image, landmarks, image)
            lines_info['fate'] = self._extract_fate_line(processed_image, landmarks, image)
            
            return lines_info
        
        except Exception as e:
            print(f"Error in line extraction: {e}")
            return {'life': None, 'heart': None, 'head': None, 'fate': None}
    
    def _create_palm_mask(self, image, landmarks):
        """
        Create a mask for the palm region using hand landmarks
        """
        try:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Define palm boundary points using key landmarks
            palm_points = np.array([
                landmarks[0],   # Wrist
                landmarks[1],   # Thumb base
                landmarks[2],   # Thumb joint
                landmarks[5],   # Index finger base
                landmarks[9],   # Middle finger base
                landmarks[13],  # Ring finger base
                landmarks[17],  # Pinky base
                # Create a more natural palm boundary
                (landmarks[17][0] - 20, landmarks[17][1] + 30),  # Extended pinky side
                (landmarks[0][0] + 30, landmarks[0][1] + 10),    # Extended wrist area
            ], dtype=np.int32)
            
            # Fill the palm region
            cv2.fillPoly(mask, [palm_points], 255)
            
            # Smooth the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            return mask
        
        except Exception as e:
            print(f"Error creating palm mask: {e}")
            h, w = image.shape[:2]
            return np.ones((h, w), dtype=np.uint8) * 255  # Return full image mask
    
    def _preprocess_for_lines(self, image, mask):
        """
        Preprocess image to enhance line detection
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply palm mask
            gray = cv2.bitwise_and(gray, mask)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Gaussian blur to reduce noise while preserving edges
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Apply unsharp masking to enhance lines
            gaussian = cv2.GaussianBlur(blurred, (9, 9), 10.0)
            unsharp_mask = cv2.addWeighted(blurred, 1.5, gaussian, -0.5, 0)
            
            return unsharp_mask
        
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    def _extract_life_line(self, image, landmarks, original_image):
        """
        Extract life line (curves around thumb from index/thumb junction to wrist)
        """
        try:
            # Life line typically runs from between thumb and index finger down towards wrist
            start_point = ((landmarks[1][0] + landmarks[5][0]) // 2, 
                          (landmarks[1][1] + landmarks[5][1]) // 2)
            end_region = landmarks[0]  # Wrist
            
            # Create region of interest for life line
            roi_points = np.array([
                landmarks[0],   # Wrist
                landmarks[1],   # Thumb base
                landmarks[2],   # Thumb joint
                landmarks[5],   # Index base
                (landmarks[5][0] - 40, landmarks[5][1] + 60),  # Extended search area
                (landmarks[0][0] + 40, landmarks[0][1])        # Extended wrist area
            ], dtype=np.int32)
            
            # Create mask for this specific region
            h, w = image.shape[:2]
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)
            
            # Apply ROI mask
            masked_image = cv2.bitwise_and(image, roi_mask)
            
            # Edge detection with multiple thresholds
            edges1 = cv2.Canny(masked_image, 30, 90)
            edges2 = cv2.Canny(masked_image, 50, 150)
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours based on position and shape
                life_line_candidates = []
                
                for contour in contours:
                    if cv2.contourArea(contour) > 50:  # Minimum area threshold
                        # Check if contour is in expected life line area
                        moments = cv2.moments(contour)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            
                            # Life line should be in the thumb-side area
                            if cx < start_point[0] + 50 and cy > start_point[1]:
                                arc_length = cv2.arcLength(contour, False)
                                life_line_candidates.append((contour, arc_length))
                
                if life_line_candidates:
                    # Select the longest contour as life line
                    life_line = max(life_line_candidates, key=lambda x: x[1])[0]
                    
                    return {
                        'contour': life_line,
                        'length': cv2.arcLength(life_line, False),
                        'area': cv2.contourArea(life_line),
                        'start_point': start_point,
                        'strength': self._calculate_line_strength(masked_image, life_line),
                        'breaks': self._detect_line_breaks(life_line),
                        'depth': self._calculate_line_depth(masked_image, life_line),
                        'curvature': self._calculate_curvature(life_line)
                    }
            
            return None
        
        except Exception as e:
            print(f"Error extracting life line: {e}")
            return None
    
    def _extract_heart_line(self, image, landmarks, original_image):
        """
        Extract heart line (runs horizontally below fingers)
        """
        try:
            # Heart line runs from pinky side towards index finger area
            start_region = landmarks[17]  # Pinky base
            end_region = landmarks[5]     # Index base
            
            # Estimate heart line position (above the palm center, below fingers)
            heart_y = landmarks[17][1] + 30  # Slightly below finger bases
            
            # Create horizontal strip ROI for heart line
            roi_height = 50
            y1 = max(0, heart_y - roi_height//2)
            y2 = min(image.shape[0], heart_y + roi_height//2)
            x1 = max(0, min(start_region[0], end_region[0]) - 30)
            x2 = min(image.shape[1], max(start_region[0], end_region[0]) + 30)
            
            if y2 <= y1 or x2 <= x1:
                return None
            
            roi = image[y1:y2, x1:x2]
            
            # Edge detection optimized for horizontal lines
            edges = cv2.Canny(roi, 20, 60)
            
            # Use Hough Line Transform to detect horizontal lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                   minLineLength=40, maxLineGap=15)
            
            if lines is not None and len(lines) > 0:
                # Select the best horizontal line
                best_line = self._select_best_horizontal_line(lines, roi.shape)
                
                if best_line is not None:
                    # Adjust coordinates back to original image space
                    best_line = (best_line[0] + x1, best_line[1] + y1, 
                               best_line[2] + x1, best_line[3] + y1)
                    
                    line_length = np.sqrt((best_line[2] - best_line[0])**2 + 
                                        (best_line[3] - best_line[1])**2)
                    
                    return {
                        'line': best_line,
                        'length': line_length,
                        'curve': self._calculate_line_curvature(best_line),
                        'position': heart_y / image.shape[0],  # Relative position
                        'strength': self._calculate_horizontal_line_strength(roi, best_line, x1, y1),
                        'breaks': self._detect_horizontal_line_breaks(roi, best_line),
                        'depth': 0.7  # Default depth for detected lines
                    }
            
            return None
        
        except Exception as e:
            print(f"Error extracting heart line: {e}")
            return None
    
    def _extract_head_line(self, image, landmarks, original_image):
        """
        Extract head line (runs horizontally in middle of palm)
        """
        try:
            # Head line typically runs from between thumb and index to opposite side
            start_x = (landmarks[1][0] + landmarks[5][0]) // 2 + 15
            start_y = (landmarks[1][1] + landmarks[5][1]) // 2 + 40
            
            # Create ROI for head line (below heart line, above wrist)
            roi_height = 40
            y1 = max(0, start_y - roi_height//2)
            y2 = min(image.shape[0], start_y + roi_height//2)
            x1 = max(0, start_x - 10)
            x2 = min(image.shape[1], landmarks[17][0] + 20)
            
            if y2 <= y1 or x2 <= x1:
                return None
            
            roi = image[y1:y2, x1:x2]
            
            # Edge detection
            edges = cv2.Canny(roi, 25, 75)
            
            # Detect lines with Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                                   minLineLength=30, maxLineGap=20)
            
            if lines is not None and len(lines) > 0:
                # Filter for mostly horizontal lines with slight slope allowed
                horizontal_lines = []
                for line in lines:
                    x1_l, y1_l, x2_l, y2_l = line[0]
                    if abs(x2_l - x1_l) > 20:  # Ensure minimum length
                        angle = abs(math.atan2(y2_l - y1_l, x2_l - x1_l) * 180 / math.pi)
                        if angle < 30:  # Allow slight slope for head line
                            length = np.sqrt((x2_l - x1_l)**2 + (y2_l - y1_l)**2)
                            horizontal_lines.append((line[0], length))
                
                if horizontal_lines:
                    # Select longest line as head line
                    best_line = max(horizontal_lines, key=lambda x: x[1])[0]
                    
                    # Adjust coordinates back to original image space
                    best_line = (best_line[0] + x1, best_line[1] + y1,
                               best_line[2] + x1, best_line[3] + y1)
                    
                    return {
                        'line': best_line,
                        'length': np.sqrt((best_line[2] - best_line[0])**2 + 
                                        (best_line[3] - best_line[1])**2),
                        'slope': self._calculate_line_slope(best_line),
                        'strength': 0.75,
                        'breaks': self._detect_horizontal_line_breaks(roi, best_line),
                        'depth': self._estimate_line_depth_from_intensity(roi, best_line, x1, y1),
                        'clarity': 0.8
                    }
            
            return None
        
        except Exception as e:
            print(f"Error extracting head line: {e}")
            return None
    
    def _extract_fate_line(self, image, landmarks, original_image):
        """
        Extract fate line (runs vertically up the palm center)
        """
        try:
            # Fate line runs from wrist towards middle finger
            palm_center_x = (landmarks[5][0] + landmarks[17][0]) // 2
            
            # Create vertical ROI for fate line
            roi_width = 60
            x1 = max(0, palm_center_x - roi_width//2)
            x2 = min(image.shape[1], palm_center_x + roi_width//2)
            y1 = max(0, landmarks[9][1] - 20)  # Above middle finger base
            y2 = min(image.shape[0], landmarks[0][1] + 20)  # Below wrist
            
            if x2 <= x1 or y2 <= y1:
                return {'presence': False, 'strength': 0}
            
            roi = image[y1:y2, x1:x2]
            
            # Edge detection optimized for vertical lines
            edges = cv2.Canny(roi, 20, 60)
            
            # Detect vertical lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                                   minLineLength=25, maxLineGap=25)
            
            if lines is not None and len(lines) > 0:
                # Filter for vertical lines
                vertical_lines = []
                for line in lines:
                    x1_l, y1_l, x2_l, y2_l = line[0]
                    if abs(y2_l - y1_l) > 15:  # Minimum vertical length
                        angle = abs(math.atan2(y2_l - y1_l, x2_l - x1_l) * 180 / math.pi)
                        if angle > 60:  # Vertical-ish lines
                            length = np.sqrt((x2_l - x1_l)**2 + (y2_l - y1_l)**2)
                            vertical_lines.append((line[0], length))
                
                if vertical_lines:
                    # Select longest vertical line as fate line
                    best_line = max(vertical_lines, key=lambda x: x[1])[0]
                    
                    # Adjust coordinates back to original image space
                    best_line = (best_line[0] + x1, best_line[1] + y1,
                               best_line[2] + x1, best_line[3] + y1)
                    
                    line_length = np.sqrt((best_line[2] - best_line[0])**2 + 
                                        (best_line[3] - best_line[1])**2)
                    
                    return {
                        'line': best_line,
                        'length': line_length,
                        'clarity': self._calculate_line_clarity(roi, best_line, x1, y1),
                        'presence': True,
                        'strength': min(line_length / 100, 1.0),  # Normalize strength
                        'breaks': self._detect_vertical_line_breaks(roi, best_line),
                        'depth': self._estimate_line_depth_from_intensity(roi, best_line, x1, y1)
                    }
            
            return {'presence': False, 'strength': 0}
        
        except Exception as e:
            print(f"Error extracting fate line: {e}")
            return {'presence': False, 'strength': 0}
    
    def _calculate_line_strength(self, image, contour):
        """Calculate line strength based on contrast and continuity"""
        try:
            if contour is None or len(contour) < 5:
                return 0.5
            
            # Create mask for the contour
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, 2)
            
            # Calculate average intensity along the line
            line_pixels = image[mask == 255]
            if len(line_pixels) > 0:
                # Stronger lines are typically darker
                avg_intensity = np.mean(line_pixels)
                strength = 1.0 - (avg_intensity / 255.0)
                return max(0.1, min(1.0, strength))
            
            return 0.5
        
        except Exception as e:
            print(f"Error calculating line strength: {e}")
            return 0.5
    
    def _detect_line_breaks(self, contour):
        """Detect breaks or gaps in a line contour"""
        try:
            if contour is None or len(contour) < 10:
                return []
            
            breaks = []
            gap_threshold = 20  # Pixels
            
            # Check distances between consecutive points
            for i in range(1, len(contour)):
                prev_point = contour[i-1][0]
                curr_point = contour[i][0]
                distance = np.sqrt((curr_point[0] - prev_point[0])**2 + 
                                 (curr_point[1] - prev_point[1])**2)
                
                if distance > gap_threshold:
                    breaks.append({
                        'position': i,
                        'gap_size': distance,
                        'location': ((prev_point[0] + curr_point[0]) // 2,
                                   (prev_point[1] + curr_point[1]) // 2)
                    })
            
            return breaks
        
        except Exception as e:
            print(f"Error detecting line breaks: {e}")
            return []
    
    def _calculate_line_depth(self, image, contour):
        """Calculate line depth based on pixel intensity"""
        try:
            if contour is None or len(contour) < 5:
                return 0.5
            
            # Create a thicker mask to sample more pixels
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, 3)
            
            line_pixels = image[mask == 255]
            if len(line_pixels) > 0:
                # Deeper lines have lower pixel values (darker)
                avg_intensity = np.mean(line_pixels)
                depth = 1.0 - (avg_intensity / 255.0)
                return max(0.1, min(1.0, depth))
            
            return 0.5
        
        except Exception as e:
            print(f"Error calculating line depth: {e}")
            return 0.5
    
    def _calculate_curvature(self, contour):
        """Calculate the curvature of a contour line"""
        try:
            if contour is None or len(contour) < 10:
                return 0.0
            
            # Convert contour to a simple array of points
            points = contour.reshape(-1, 2)
            
            # Calculate curvature using change in direction
            total_curvature = 0
            for i in range(2, len(points) - 2):
                p1, p2, p3 = points[i-1], points[i], points[i+1]
                
                # Calculate vectors
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Calculate angle between vectors
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    total_curvature += abs(angle)
            
            # Normalize by number of segments
            avg_curvature = total_curvature / max(len(points) - 4, 1)
            return min(avg_curvature, 2.0)  # Cap at reasonable value
        
        except Exception as e:
            print(f"Error calculating curvature: {e}")
            return 0.0
    
    def _select_best_horizontal_line(self, lines, roi_shape):
        """Select the best horizontal line from detected lines"""
        try:
            if len(lines) == 0:
                return None
            
            best_line = None
            best_score = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Calculate horizontalness (prefer horizontal lines)
                if abs(x2 - x1) > 0:
                    angle = abs(math.atan2(y2-y1, x2-x1) * 180 / math.pi)
                    horizontal_score = 1.0 - (angle / 90.0)  # Higher score for more horizontal
                else:
                    horizontal_score = 0
                
                # Position score (prefer lines in middle of ROI vertically)
                y_center = (y1 + y2) / 2
                roi_center = roi_shape[0] / 2
                position_score = 1.0 - abs(y_center - roi_center) / roi_center
                
                # Combined score
                score = length * horizontal_score * position_score
                
                if score > best_score:
                    best_score = score
                    best_line = line[0]
            
            return best_line
        
        except Exception as e:
            print(f"Error selecting best horizontal line: {e}")
            return None
    
    def _calculate_line_curvature(self, line):
        """Calculate curvature of a straight line (returns small value for straight lines)"""
        try:
            if line is None:
                return 0
            
            x1, y1, x2, y2 = line
            # For straight lines, curvature is based on deviation from horizontal
            if abs(x2 - x1) > 0:
                slope = abs(y2 - y1) / abs(x2 - x1)
                return min(slope, 1.0)  # Normalize to 0-1 range
            return 0
        
        except Exception as e:
            print(f"Error calculating line curvature: {e}")
            return 0
    
    def _calculate_line_slope(self, line):
        """Calculate slope of a line"""
        try:
            if line is None:
                return 0
            
            x1, y1, x2, y2 = line
            if abs(x2 - x1) > 0:
                return (y2 - y1) / (x2 - x1)
            return float('inf') if y2 != y1 else 0
        
        except Exception as e:
            print(f"Error calculating line slope: {e}")
            return 0
    
    def _calculate_horizontal_line_strength(self, roi, line, offset_x, offset_y):
        """Calculate strength of a horizontal line"""
        try:
            if line is None:
                return 0.5
            
            x1, y1, x2, y2 = line
            # Adjust for ROI coordinates
            x1_adj = x1 - offset_x if offset_x <= x1 else 0
            x2_adj = x2 - offset_x if offset_x <= x2 else roi.shape[1] - 1
            y_adj = y1 - offset_y if offset_y <= y1 else 0
            
            # Sample pixels along the line
            if x2_adj > x1_adj and 0 <= y_adj < roi.shape[0]:
                line_pixels = roi[y_adj, x1_adj:x2_adj+1]
                if len(line_pixels) > 0:
                    avg_intensity = np.mean(line_pixels)
                    return 1.0 - (avg_intensity / 255.0)
            
            return 0.5
        
        except Exception as e:
            print(f"Error calculating horizontal line strength: {e}")
            return 0.5
    
    def _detect_horizontal_line_breaks(self, roi, line):
        """Detect breaks in a horizontal line"""
        try:
            if line is None:
                return []
            
            x1, y1, x2, y2 = line
            if x2 <= x1 or y1 < 0 or y1 >= roi.shape[0]:
                return []
            
            # Sample pixels along the line
            line_pixels = roi[y1, x1:x2+1]
            breaks = []
            
            # Find bright spots (gaps) in the line
            threshold = np.mean(line_pixels) + np.std(line_pixels)
            
            in_gap = False
            gap_start = None
            
            for i, pixel in enumerate(line_pixels):
                if pixel > threshold:  # Bright pixel (potential gap)
                    if not in_gap:
                        gap_start = i + x1
                        in_gap = True
                else:  # Dark pixel (line continues)
                    if in_gap:
                        gap_end = i + x1
                        breaks.append({
                            'start': gap_start,
                            'end': gap_end,
                            'length': gap_end - gap_start,
                            'position': (gap_start + gap_end) // 2
                        })
                        in_gap = False
            
            return breaks
        
        except Exception as e:
            print(f"Error detecting horizontal line breaks: {e}")
            return []
    
    def _detect_vertical_line_breaks(self, roi, line):
        """Detect breaks in a vertical line"""
        try:
            if line is None:
                return []
            
            x1, y1, x2, y2 = line
            if y2 <= y1 or x1 < 0 or x1 >= roi.shape[1]:
                return []
            
            # Sample pixels along the vertical line
            line_pixels = roi[y1:y2+1, x1]
            breaks = []
            
            if len(line_pixels) == 0:
                return breaks
            
            # Find bright spots (gaps) in the line
            threshold = np.mean(line_pixels) + np.std(line_pixels)
            
            in_gap = False
            gap_start = None
            
            for i, pixel in enumerate(line_pixels):
                if pixel > threshold:  # Bright pixel (potential gap)
                    if not in_gap:
                        gap_start = i + y1
                        in_gap = True
                else:  # Dark pixel (line continues)
                    if in_gap:
                        gap_end = i + y1
                        breaks.append({
                            'start': gap_start,
                            'end': gap_end,
                            'length': gap_end - gap_start,
                            'position': (gap_start + gap_end) // 2
                        })
                        in_gap = False
            
            return breaks
        
        except Exception as e:
            print(f"Error detecting vertical line breaks: {e}")
            return []
    
    def _calculate_line_clarity(self, roi, line, offset_x, offset_y):
        """Calculate clarity/definition of a line"""
        try:
            if line is None:
                return 0.5
            
            x1, y1, x2, y2 = line
            
            # Adjust coordinates for ROI offset
            x1_adj = max(0, x1 - offset_x)
            x2_adj = min(roi.shape[1] - 1, x2 - offset_x)
            y1_adj = max(0, y1 - offset_y)
            y2_adj = min(roi.shape[0] - 1, y2 - offset_y)
            
            # Create line mask
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.line(mask, (x1_adj, y1_adj), (x2_adj, y2_adj), 255, 3)
            
            # Calculate gradient magnitude along line
            sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            line_gradient = gradient_magnitude[mask == 255]
            if len(line_gradient) > 0:
                clarity = np.mean(line_gradient) / 255.0
                return max(0.1, min(1.0, clarity))
            
            return 0.5
        
        except Exception as e:
            print(f"Error calculating line clarity: {e}")
            return 0.5
    
    def _estimate_line_depth_from_intensity(self, roi, line, offset_x, offset_y):
        """Estimate line depth from pixel intensity"""
        try:
            if line is None:
                return 0.5
            
            x1, y1, x2, y2 = line
            
            # Adjust coordinates for ROI
            x1_adj = max(0, x1 - offset_x)
            x2_adj = min(roi.shape[1] - 1, x2 - offset_x)
            y1_adj = max(0, y1 - offset_y)
            y2_adj = min(roi.shape[0] - 1, y2 - offset_y)
            
            # Sample pixels along the line
            num_samples = 20
            line_pixels = []
            
            for i in range(num_samples):
                t = i / (num_samples - 1)
                x = int(x1_adj + t * (x2_adj - x1_adj))
                y = int(y1_adj + t * (y2_adj - y1_adj))
                
                if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                    line_pixels.append(roi[y, x])
            
            if line_pixels:
                avg_intensity = np.mean(line_pixels)
                depth = 1.0 - (avg_intensity / 255.0)  # Darker = deeper
                return max(0.1, min(1.0, depth))
            
            return 0.5
        
        except Exception as e:
            print(f"Error estimating line depth: {e}")
            return 0.5
    
    def enhance_line_visualization(self, image, lines_info, landmarks):
        """
        Create an enhanced visualization of detected lines
        """
        try:
            enhanced = image.copy()
            
            # Draw detected lines with different colors
            for line_type, line_data in lines_info.items():
                if line_data is None:
                    continue
                
                color = self.line_colors.get(line_type, (255, 255, 255))
                
                if line_type == 'life' and 'contour' in line_data:
                    # Draw life line contour
                    cv2.drawContours(enhanced, [line_data['contour']], -1, color, 3)
                    
                    # Mark breaks if any
                    breaks = line_data.get('breaks', [])
                    for break_info in breaks:
                        if isinstance(break_info, dict) and 'location' in break_info:
                            cv2.circle(enhanced, break_info['location'], 5, (0, 0, 255), -1)
                
                elif 'line' in line_data:
                    # Draw straight line
                    line = line_data['line']
                    cv2.line(enhanced, (line[0], line[1]), (line[2], line[3]), color, 3)
                    
                    # Add line label
                    mid_x = (line[0] + line[2]) // 2
                    mid_y = (line[1] + line[3]) // 2
                    cv2.putText(enhanced, line_type.title(), (mid_x, mid_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return enhanced
        
        except Exception as e:
            print(f"Error enhancing line visualization: {e}")
            return image
    
    def get_line_statistics(self, lines_info):
        """
        Generate statistical summary of detected lines
        """
        try:
            stats = {
                'total_lines_detected': 0,
                'average_line_strength': 0,
                'total_breaks_detected': 0,
                'line_details': {}
            }
            
            total_strength = 0
            valid_lines = 0
            
            for line_type, line_data in lines_info.items():
                if line_data is None:
                    stats['line_details'][line_type] = {'detected': False}
                    continue
                
                line_stats = {'detected': True}
                
                # Extract common statistics
                if 'strength' in line_data:
                    line_stats['strength'] = line_data['strength']
                    total_strength += line_data['strength']
                    valid_lines += 1
                
                if 'length' in line_data:
                    line_stats['length'] = line_data['length']
                
                if 'breaks' in line_data:
                    num_breaks = len(line_data['breaks'])
                    line_stats['breaks_count'] = num_breaks
                    stats['total_breaks_detected'] += num_breaks
                
                if 'depth' in line_data:
                    line_stats['depth'] = line_data['depth']
                
                stats['line_details'][line_type] = line_stats
                stats['total_lines_detected'] += 1
            
            # Calculate average strength
            if valid_lines > 0:
                stats['average_line_strength'] = total_strength / valid_lines
            
            return stats
        
        except Exception as e:
            print(f"Error generating line statistics: {e}")
            return {'total_lines_detected': 0, 'average_line_strength': 0}