import json
import os
import numpy as np

class PalmistryAnalyzer:
    def __init__(self):
        self.meanings = self._load_palmistry_meanings()
    
    def _load_palmistry_meanings(self):
        """Load palmistry interpretations from JSON file"""
        try:
            data_path = os.path.join('data', 'palmistry_meanings.json')
            with open(data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default meanings if file not found
            return self._get_default_meanings()
    
    def _get_default_meanings(self):
        """Default palmistry meanings"""
        return {
            "life_line": {
                "long_deep": "Strong vitality, robust health, long life potential",
                "short_clear": "Intense but focused life energy, quality over quantity",
                "broken": "Major life changes, health challenges to overcome",
                "chained": "Health fluctuations, need for self-care attention",
                "curved": "Adventurous spirit, love for travel and new experiences"
            },
            "heart_line": {
                "long_curved": "Romantic nature, expressive emotions, warm relationships",
                "short_straight": "Reserved in love, practical approach to relationships",
                "ends_under_index": "High standards in love, selective in partnerships",
                "ends_under_middle": "Self-focused in relationships, independent nature",
                "broken": "Emotional trauma, heartbreak recovery needed"
            },
            "head_line": {
                "long_straight": "Logical thinking, analytical mind, practical approach",
                "long_curved": "Creative intelligence, imaginative problem-solving",
                "short": "Quick decisions, prefer action over prolonged thinking",
                "sloped": "Creative and artistic inclinations, intuitive nature",
                "broken": "Mental growth phases, changing thought patterns"
            },
            "fate_line": {
                "present_clear": "Strong sense of destiny, clear life direction",
                "absent": "Self-made path, freedom from predetermined destiny",
                "starts_late": "Late bloomer, major success after age 35",
                "multiple": "Multiple career paths, diverse interests and talents",
                "broken": "Career changes, varied professional journey"
            },
            "hand_shapes": {
                "square": "Practical, reliable, methodical, traditional values",
                "pointed": "Intuitive, artistic, sensitive, idealistic nature",
                "spatulate": "Energetic, adventurous, innovative, active lifestyle",
                "conic": "Balanced personality, diplomatic, adaptable nature"
            }
        }
    
    def analyze_palm(self, lines_info, landmarks):
        """
        Main analysis function that interprets palm features
        """
        analysis = {
            'life_line': self._analyze_life_line(lines_info.get('life')),
            'heart_line': self._analyze_heart_line(lines_info.get('heart')),
            'head_line': self._analyze_head_line(lines_info.get('head')),
            'fate_line': self._analyze_fate_line(lines_info.get('fate')),
            'hand_shape': self._analyze_hand_shape(landmarks),
            'overall_summary': '',
            'personality_scores': {},
            'life_aspects': {}
        }
        
        # Generate personality scores
        analysis['personality_scores'] = self._calculate_personality_scores(analysis)
        
        # Generate life aspects analysis
        analysis['life_aspects'] = self._analyze_life_aspects(analysis)
        
        # Create overall summary
        analysis['overall_summary'] = self._generate_overall_summary(analysis)
        
        return analysis
    
    def _analyze_life_line(self, life_line_info):
        """Analyze life line characteristics"""
        if not life_line_info:
            return {
                'interpretation': 'Life line not clearly detected',
                'characteristics': ['Unclear reading'],
                'score': 50,
                'details': 'Unable to analyze life line from the image'
            }
        
        characteristics = []
        score = 70  # Base score
        
        # Analyze length
        length = life_line_info.get('length', 0)
        if length > 200:
            characteristics.append('Long life line - strong vitality')
            score += 15
            interpretation = self.meanings['life_line']['long_deep']
        elif length > 100:
            characteristics.append('Moderate life line - balanced energy')
            score += 5
            interpretation = 'Balanced life energy with steady vitality'
        else:
            characteristics.append('Short life line - intense energy')
            interpretation = self.meanings['life_line']['short_clear']
        
        # Analyze depth/strength
        strength = life_line_info.get('strength', 0.5)
        if strength > 0.7:
            characteristics.append('Deep line - robust health')
            score += 10
        elif strength < 0.3:
            characteristics.append('Faint line - need for health attention')
            score -= 5
        
        # Analyze breaks
        breaks = life_line_info.get('breaks', [])
        if len(breaks) > 0:
            characteristics.append(f'Line breaks - {len(breaks)} major life changes')
            interpretation = self.meanings['life_line']['broken']
            score -= len(breaks) * 5
        
        return {
            'interpretation': interpretation,
            'characteristics': characteristics,
            'score': min(max(score, 0), 100),
            'details': f'Length: {length:.0f}px, Strength: {strength:.2f}, Breaks: {len(breaks)}'
        }
    
    def _analyze_heart_line(self, heart_line_info):
        """Analyze heart line characteristics"""
        if not heart_line_info:
            return {
                'interpretation': 'Heart line not clearly detected',
                'characteristics': ['Unclear emotional reading'],
                'score': 50,
                'details': 'Unable to analyze heart line from the image'
            }
        
        characteristics = []
        score = 70
        
        # Analyze length and position
        length = heart_line_info.get('length', 0)
        position = heart_line_info.get('position', 0.5)
        
        if length > 150:
            characteristics.append('Long heart line - expressive emotions')
            score += 10
            interpretation = self.meanings['heart_line']['long_curved']
        elif length < 80:
            characteristics.append('Short heart line - reserved emotions')
            interpretation = self.meanings['heart_line']['short_straight']
        else:
            characteristics.append('Moderate heart line - balanced emotions')
            interpretation = 'Balanced emotional expression and relationships'
        
        # Analyze curve
        curve = heart_line_info.get('curve', 0)
        if curve > 0.3:
            characteristics.append('Curved line - warm and romantic')
            score += 10
        elif curve < 0.1:
            characteristics.append('Straight line - practical in love')
            score += 5
        
        # Analyze breaks
        breaks = heart_line_info.get('breaks', [])
        if len(breaks) > 0:
            characteristics.append('Broken line - emotional challenges')
            interpretation = self.meanings['heart_line']['broken']
            score -= 10
        
        return {
            'interpretation': interpretation,
            'characteristics': characteristics,
            'score': min(max(score, 0), 100),
            'details': f'Length: {length:.0f}px, Curve: {curve:.2f}, Position: {position:.2f}'
        }
    
    def _analyze_head_line(self, head_line_info):
        """Analyze head line characteristics"""
        if not head_line_info:
            return {
                'interpretation': 'Head line not clearly detected',
                'characteristics': ['Unclear mental reading'],
                'score': 50,
                'details': 'Unable to analyze head line from the image'
            }
        
        characteristics = []
        score = 70
        
        # Analyze length
        length = head_line_info.get('length', 0)
        if length > 120:
            characteristics.append('Long head line - analytical thinking')
            score += 10
            interpretation = self.meanings['head_line']['long_straight']
        elif length < 60:
            characteristics.append('Short head line - quick decisions')
            interpretation = self.meanings['head_line']['short']
        else:
            characteristics.append('Moderate head line - balanced thinking')
            interpretation = 'Balanced analytical and intuitive thinking'
        
        # Analyze slope
        slope = abs(head_line_info.get('slope', 0))
        if slope > 0.3:
            characteristics.append('Sloped line - creative and imaginative')
            interpretation = self.meanings['head_line']['sloped']
            score += 15
        elif slope < 0.1:
            characteristics.append('Straight line - logical and practical')
            score += 10
        
        # Analyze breaks
        breaks = head_line_info.get('breaks', [])
        if len(breaks) > 0:
            characteristics.append('Line breaks - changing thought patterns')
            interpretation = self.meanings['head_line']['broken']
        
        return {
            'interpretation': interpretation,
            'characteristics': characteristics,
            'score': min(max(score, 0), 100),
            'details': f'Length: {length:.0f}px, Slope: {slope:.2f}, Breaks: {len(breaks)}'
        }
    
    def _analyze_fate_line(self, fate_line_info):
        """Analyze fate line characteristics"""
        if not fate_line_info or not fate_line_info.get('presence', False):
            return {
                'interpretation': self.meanings['fate_line']['absent'],
                'characteristics': ['No clear fate line - self-made destiny'],
                'score': 75,
                'details': 'Absence of fate line indicates freedom and self-determination'
            }
        
        characteristics = []
        score = 80
        
        # Analyze clarity
        clarity = fate_line_info.get('clarity', 0.5)
        if clarity > 0.7:
            characteristics.append('Clear fate line - strong destiny sense')
            interpretation = self.meanings['fate_line']['present_clear']
            score += 15
        elif clarity > 0.4:
            characteristics.append('Moderate fate line - some life direction')
            interpretation = 'Moderate sense of life direction and purpose'
            score += 5
        else:
            characteristics.append('Faint fate line - flexible life path')
            interpretation = 'Flexible approach to life direction'
        
        # Analyze length and position
        length = fate_line_info.get('length', 0)
        if length > 100:
            characteristics.append('Long fate line - lifelong purpose')
            score += 10
        
        return {
            'interpretation': interpretation,
            'characteristics': characteristics,
            'score': min(max(score, 0), 100),
            'details': f'Clarity: {clarity:.2f}, Length: {length:.0f}px'
        }
    
    def _analyze_hand_shape(self, landmarks):
        """Analyze hand shape and its meaning"""
        if not landmarks or len(landmarks) < 21:
            return {
                'type': 'unknown',
                'interpretation': 'Unable to determine hand shape',
                'characteristics': ['Hand shape unclear'],
                'score': 50
            }
        
        # Calculate hand proportions
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        index_base = landmarks[5]
        pinky_base = landmarks[17]
        
        palm_width = abs(index_base[0] - pinky_base[0])
        palm_length = abs(landmarks[9][1] - wrist[1])
        finger_length = abs(middle_tip[1] - landmarks[9][1])
        
        if palm_width == 0 or palm_length == 0:
            return {
                'type': 'unknown',
                'interpretation': 'Unable to calculate hand proportions',
                'characteristics': ['Measurement error'],
                'score': 50
            }
        
        palm_ratio = palm_length / palm_width
        finger_palm_ratio = finger_length / palm_length if palm_length > 0 else 1
        
        # Determine hand shape
        if palm_ratio < 0.85:
            shape_type = 'square'
        elif palm_ratio > 1.15:
            if finger_palm_ratio > 0.9:
                shape_type = 'pointed'
            else:
                shape_type = 'conic'
        elif finger_palm_ratio < 0.75:
            shape_type = 'spatulate'
        else:
            shape_type = 'conic'
        
        interpretation = self.meanings['hand_shapes'].get(shape_type, 'Balanced personality traits')
        
        characteristics = [f'{shape_type.title()} hand shape']
        if shape_type == 'square':
            characteristics.append('Practical and methodical nature')
        elif shape_type == 'pointed':
            characteristics.append('Artistic and intuitive tendencies')
        elif shape_type == 'spatulate':
            characteristics.append('Energetic and adventurous spirit')
        elif shape_type == 'conic':
            characteristics.append('Diplomatic and adaptable personality')
        
        return {
            'type': shape_type,
            'interpretation': interpretation,
            'characteristics': characteristics,
            'score': 85,
            'details': f'Palm ratio: {palm_ratio:.2f}, Finger ratio: {finger_palm_ratio:.2f}'
        }
    
    def _calculate_personality_scores(self, analysis):
        """Calculate personality trait percentages"""
        scores = {}
        
        # Creativity score
        creativity = 60  # Base score
        if 'creative' in analysis['head_line']['interpretation'].lower():
            creativity += 20
        if 'artistic' in analysis['hand_shape']['interpretation'].lower():
            creativity += 15
        if analysis['head_line']['score'] > 80:
            creativity += 10
        
        # Leadership score
        leadership = 50
        if analysis['fate_line']['score'] > 75:
            leadership += 20
        if 'strong' in analysis['life_line']['interpretation'].lower():
            leadership += 15
        if analysis['hand_shape']['type'] == 'square':
            leadership += 10
        
        # Emotional Intelligence
        emotional = 55
        if analysis['heart_line']['score'] > 80:
            emotional += 25
        if 'expressive' in analysis['heart_line']['interpretation'].lower():
            emotional += 15
        
        # Analytical Thinking
        analytical = 60
        if 'logical' in analysis['head_line']['interpretation'].lower():
            analytical += 20
        if analysis['hand_shape']['type'] == 'square':
            analytical += 15
        if analysis['head_line']['score'] > 85:
            analytical += 10
        
        # Intuition
        intuition = 50
        if 'intuitive' in analysis['hand_shape']['interpretation'].lower():
            intuition += 25
        if 'curved' in analysis['head_line']['interpretation'].lower():
            intuition += 15
        
        # Health Vitality
        vitality = analysis['life_line']['score']
        
        scores = {
            'creativity': min(creativity, 100),
            'leadership': min(leadership, 100),
            'emotional_intelligence': min(emotional, 100),
            'analytical_thinking': min(analytical, 100),
            'intuition': min(intuition, 100),
            'vitality': min(vitality, 100)
        }
        
        return scores
    
    def _analyze_life_aspects(self, analysis):
        """Analyze different life aspects"""
        aspects = {}
        
        # Career and Success
        career_score = (analysis['fate_line']['score'] + analysis['head_line']['score']) / 2
        if career_score > 80:
            career_outlook = 'Strong career potential with clear direction'
        elif career_score > 60:
            career_outlook = 'Good career prospects with some planning needed'
        else:
            career_outlook = 'Career path requires focus and determination'
        
        # Love and Relationships
        love_score = analysis['heart_line']['score']
        if love_score > 80:
            love_outlook = 'Rich emotional life with meaningful relationships'
        elif love_score > 60:
            love_outlook = 'Balanced approach to love and relationships'
        else:
            love_outlook = 'Growth needed in emotional expression and relationships'
        
        # Health and Wellness
        health_score = analysis['life_line']['score']
        if health_score > 80:
            health_outlook = 'Strong vitality and robust health potential'
        elif health_score > 60:
            health_outlook = 'Good health with attention to lifestyle balance'
        else:
            health_outlook = 'Focus on health and wellness is important'
        
        aspects = {
            'career': {
                'score': int(career_score),
                'outlook': career_outlook
            },
            'love': {
                'score': int(love_score),
                'outlook': love_outlook
            },
            'health': {
                'score': int(health_score),
                'outlook': health_outlook
            }
        }
        
        return aspects
    
    def _generate_overall_summary(self, analysis):
        """Generate a comprehensive summary"""
        hand_shape = analysis['hand_shape']['type']
        
        # Start with hand shape personality
        if hand_shape == 'square':
            base_summary = "You have a practical and methodical approach to life. "
        elif hand_shape == 'pointed':
            base_summary = "You possess an artistic and intuitive nature. "
        elif hand_shape == 'spatulate':
            base_summary = "You are energetic and adventurous by nature. "
        else:
            base_summary = "You have a balanced and adaptable personality. "
        
        # Add dominant traits based on highest scores
        scores = analysis['personality_scores']
        top_trait = max(scores.items(), key=lambda x: x[1])
        
        if top_trait[1] > 80:
            if top_trait[0] == 'creativity':
                base_summary += "Your creative abilities are particularly strong and should be nurtured. "
            elif top_trait[0] == 'leadership':
                base_summary += "You have natural leadership qualities that can guide you to success. "
            elif top_trait[0] == 'emotional_intelligence':
                base_summary += "Your emotional intelligence is a key strength in relationships. "
            elif top_trait[0] == 'analytical_thinking':
                base_summary += "Your analytical mind is well-suited for problem-solving and planning. "
            elif top_trait[0] == 'intuition':
                base_summary += "Trust your intuition as it's one of your strongest guides. "
        
        # Add life aspect insights
        career_score = analysis['life_aspects']['career']['score']
        love_score = analysis['life_aspects']['love']['score']
        
        if career_score > love_score and career_score > 75:
            base_summary += "Career success is well-indicated in your palm. "
        elif love_score > 75:
            base_summary += "Meaningful relationships play a central role in your happiness. "
        
        base_summary += "Focus on developing your natural strengths while being mindful of areas that need attention."
        
        return base_summary