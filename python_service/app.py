import os
import pickle
import re
import nltk
import json
import warnings
from textblob import TextBlob
from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.exceptions import InconsistentVersionWarning
import logging
import io
# Optional imports with fallbacks
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except (ImportError, OSError):
    spacy = None
    nlp = None
    print("Warning: spaCy not available, using fallback NER")

try:
    from polyglot.detect import Detector
    from polyglot.text import Text
    POLYGLOT_AVAILABLE = True
except ImportError:
    POLYGLOT_AVAILABLE = False
    print("Warning: Polyglot not available, using langdetect only")

import logging
import io
import csv
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# --- Suppress sklearn version warnings ---
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- Setup Logging ---
logging.basicConfig(filename='disaster_errors.log', level=logging.ERROR, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download required NLTK data (run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize language detection and translation
DetectorFactory.seed = 0  # For consistent language detection

# Initialize sentiment analysis
try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    logging.error(f"Failed to load NLP tools: {e}")
    print(f"CRITICAL ERROR: Failed to load NLP tools. Error: {e}")
    raise

# Initialize spaCy for enhanced NER
# spaCy initialization moved to imports section

def load_json_file(file_path, default_data={}):
    """Helper function to load JSON files with robust error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}. Using default data.")
        print(f"Warning: {file_path} not found. Using default data.")
        return default_data
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in {file_path}. Using default data.")
        print(f"Warning: {file_path} is not a valid JSON file. Using default data.")
        return default_data

# Enhanced disaster keywords with better categorization
disaster_keywords = load_json_file('disaster_keywords.txt', {
    "earthquake": ["earthquake", "quake", "tremor", "seismic", "aftershock", "epicenter"],
    "flood": ["flood", "flooding", "inundation", "deluge", "overflow", "waterlogged"],
    "hurricane": ["hurricane", "cyclone", "typhoon", "storm surge", "tropical storm"],
    "tornado": ["tornado", "twister", "funnel cloud", "wind damage"],
    "wildfire": ["wildfire", "forest fire", "blaze", "brush fire", "bushfire"],
    "tsunami": ["tsunami", "tidal wave", "seismic wave"],
    "volcanic_eruption": ["volcano", "eruption", "lava", "volcanic ash", "magma"],
    "landslide": ["landslide", "mudslide", "rockslide", "slope failure"],
    "drought": ["drought", "water shortage", "dry spell", "arid"],
    "heatwave": ["heatwave", "extreme heat", "heat stroke", "scorching"]
})

# Enhanced Indian location aliases and cities
city_aliases = load_json_file('city_aliases.json', {
    # International aliases
    'ny': 'New York', 'nyc': 'New York', 'la': 'Los Angeles', 'chi': 'Chicago',
    'sf': 'San Francisco', 'vegas': 'Las Vegas', 'ldn': 'London',
    
    # Indian cities and aliases
    'mumbai': 'Mumbai', 'mum': 'Mumbai', 'bombay': 'Mumbai',
    'delhi': 'Delhi', 'new delhi': 'New Delhi', 'ncr': 'Delhi NCR',
    'bangalore': 'Bangalore', 'bengaluru': 'Bangalore', 'blr': 'Bangalore',
    'chennai': 'Chennai', 'madras': 'Chennai', 'maa': 'Chennai',
    'kolkata': 'Kolkata', 'calcutta': 'Kolkata', 'ccu': 'Kolkata',
    'hyderabad': 'Hyderabad', 'hyd': 'Hyderabad', 'cyberabad': 'Hyderabad',
    'pune': 'Pune', 'poona': 'Pune',
    'ahmedabad': 'Ahmedabad', 'amdavad': 'Ahmedabad',
    'kochi': 'Kochi', 'cochin': 'Kochi', 'ernakulam': 'Kochi',
    'thiruvananthapuram': 'Thiruvananthapuram', 'trivandrum': 'Thiruvananthapuram',
    'bhubaneswar': 'Bhubaneswar', 'bbsr': 'Bhubaneswar',
    'guwahati': 'Guwahati', 'ghy': 'Guwahati',
    
    # Indian states
    'kerala': 'Kerala', 'tamil nadu': 'Tamil Nadu', 'tn': 'Tamil Nadu',
    'karnataka': 'Karnataka', 'ka': 'Karnataka',
    'andhra pradesh': 'Andhra Pradesh', 'ap': 'Andhra Pradesh',
    'telangana': 'Telangana', 'ts': 'Telangana',
    'maharashtra': 'Maharashtra', 'mh': 'Maharashtra',
    'gujarat': 'Gujarat', 'gj': 'Gujarat',
    'rajasthan': 'Rajasthan', 'rj': 'Rajasthan',
    'west bengal': 'West Bengal', 'wb': 'West Bengal',
    'odisha': 'Odisha', 'orissa': 'Odisha',
    'assam': 'Assam', 'as': 'Assam',
    'punjab': 'Punjab', 'pb': 'Punjab',
    'haryana': 'Haryana', 'hr': 'Haryana',
    'uttarakhand': 'Uttarakhand', 'uk': 'Uttarakhand',
    'himachal pradesh': 'Himachal Pradesh', 'hp': 'Himachal Pradesh'
})

# Indian locations for enhanced NER
INDIAN_LOCATIONS = [
    'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 
    'Ahmedabad', 'Kochi', 'Thiruvananthapuram', 'Bhubaneswar', 'Guwahati',
    'Kerala', 'Tamil Nadu', 'Karnataka', 'Andhra Pradesh', 'Telangana',
    'Maharashtra', 'Gujarat', 'Rajasthan', 'West Bengal', 'Odisha', 'Assam',
    'Punjab', 'Haryana', 'Uttarakhand', 'Himachal Pradesh', 'Uttar Pradesh',
    'Madhya Pradesh', 'Chhattisgarh', 'Jharkhand', 'Bihar', 'Tripura',
    'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Arunachal Pradesh',
    'Sikkim', 'Goa', 'Jammu and Kashmir', 'Ladakh'
]

# Enhanced false positive filters with improved contextual patterns
FALSE_POSITIVE_PATTERNS = [
    r'\b(exam|test|quiz|assignment|homework|project)\s+(?:is\s+a\s+)?disaster\b',
    r'\b(movie|film|show|series|book)\s+(?:is\s+a\s+)?disaster\b',
    r'\b(cooking|recipe|food|kitchen)\s+disaster\b',
    r'\b(hair|makeup|fashion|outfit)\s+disaster\b',
    r'\b(date|relationship|marriage)\s+disaster\b',
    r'\b(traffic|commute|journey)\s+disaster\b',
    r'\b(weather|rain|snow)\s+(?:looks|seems|appears)\s+(?:like\s+a\s+)?disaster\b',
    r'\barea\s+(?:is\s+)?(?:flooded|packed|filled|crowded)\s+with\s+people\b',
    r'\bflooded\s+with\s+(?:people|crowd|fans|supporters|visitors)\b',
    r'\b(?:totally|completely|absolutely)\s+destroyed\s+(?:that|this|my|his|her)\b',
    r'\b(?:my|his|her|the|this|that)\s+(?:life|day|week|weekend|vacation|trip)\s+(?:is|was)\s+(?:a\s+)?disaster\b'
]

# Contextual indicators for real disasters
DISASTER_CONTEXT_INDICATORS = [
    # Urgency and action words
    "help", "emergency", "urgent", "rescue", "evacuation", "evacuate", "alert", "warning",
    "trapped", "stranded", "missing", "injured", "casualties", "victims", "affected",
    
    # Impact descriptors
    "destroyed", "damaged", "collapsed", "hit", "struck", "devastated", "affected",
    "widespread", "severe", "massive", "major", "critical", "serious",
    
    # Response and reporting terms
    "reported", "confirmed", "breaking", "live", "happening", "occurring",
    "response", "relief", "aid", "support", "assistance",
    
    # Time indicators
    "just", "now", "currently", "ongoing", "breaking", "today", "yesterday"
]

# Location-only patterns (should NOT be disasters)
LOCATION_ONLY_PATTERNS = [
    r'^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*$',  # Just location names
    r'^\s*(?:in|at|from|to)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*$'  # Preposition + location
]


def clean_tweet(tweet):
    """Cleans a tweet by removing URLs, hashtags, mentions, and special characters."""
    if not tweet or not isinstance(tweet, str):
        return 'Empty'
    
    # Remove URLs
    tweet = re.sub(r'https?://\S+', '', tweet)
    tweet = re.sub(r'www\.\S+', '', tweet)
    
    # Remove hashtags but keep the text
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    
    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove special characters and excessive punctuation
    tweet = re.sub(r'[*%!]{2,}', '', tweet)
    tweet = re.sub(r'[^\w\s.,!?-]', ' ', tweet)
    
    # Clean up whitespace
    cleaned = ' '.join(tweet.split()).strip()
    return cleaned if cleaned else 'Empty'

def detect_language(tweet):
    """Enhanced language detection using langdetect with fallback to TextBlob."""
    try:
        if len(tweet.strip()) < 3:
            return 'en'
        
        # Primary detection using langdetect
        detected = detect(tweet)
        
        # Map common language codes
        language_mapping = {
            'hi': 'hindi',
            'bn': 'bengali', 
            'te': 'telugu',
            'ta': 'tamil',
            'mr': 'marathi',
            'gu': 'gujarati',
            'kn': 'kannada',
            'ml': 'malayalam',
            'pa': 'punjabi',
            'or': 'odia',
            'as': 'assamese',
            'ur': 'urdu'
        }
        
        return language_mapping.get(detected, detected)
        
    except Exception as e:
        # Fallback to TextBlob
        try:
            blob = TextBlob(tweet)
            detected = blob.detect_language()
            return detected if detected else 'en'
        except Exception:
            logging.warning(f"Language detection failed for tweet: '{tweet}'. Error: {e}")
            return 'en'

def translate_tweet(tweet, source_lang):
    """Enhanced translation using deep-translator with fallback to TextBlob."""
    if source_lang == 'en' or not tweet or len(tweet.strip()) < 3:
        return tweet
    
    try:
        # Primary translation using deep-translator
        translator = GoogleTranslator(source='auto', target='en')
        translated = translator.translate(tweet)
        return translated if translated else tweet
    except Exception as e:
        # Fallback to TextBlob
        try:
            blob = TextBlob(tweet)
            translated = blob.translate(to='en')
            return str(translated) if translated else tweet
        except Exception:
            logging.error(f"Translation error for tweet: '{tweet}'. Error: {e}")
            return tweet

def is_location_only(tweet):
    """Check if tweet is just a location name without disaster context."""
    tweet_clean = tweet.strip()
    
    # Check for location-only patterns
    for pattern in LOCATION_ONLY_PATTERNS:
        if re.match(pattern, tweet_clean, re.IGNORECASE):
            return True
    
    # Check if it's just a country/state/city name
    words = tweet_clean.split()
    if len(words) <= 2:
        # Check if all words are locations
        all_locations = True
        for word in words:
            if word.lower() not in [loc.lower() for loc in INDIAN_LOCATIONS + list(city_aliases.keys())]:
                all_locations = False
                break
        if all_locations:
            return True
    
    return False

def has_disaster_context(tweet):
    """Check if tweet has disaster-related contextual indicators."""
    tweet_lower = tweet.lower()
    
    # Count contextual indicators
    context_count = sum(1 for indicator in DISASTER_CONTEXT_INDICATORS 
                       if indicator.lower() in tweet_lower)
    
    # Check for disaster keywords with better matching
    has_keywords = False
    for keywords in disaster_keywords.values():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', tweet_lower):
                has_keywords = True
                break
        if has_keywords:
            break
    
    # More lenient context detection
    return context_count >= 1 or has_keywords

def is_false_positive(tweet):
    """Enhanced false positive detection with contextual analysis."""
    tweet_lower = tweet.lower()
    
    # Check existing false positive patterns
    for pattern in FALSE_POSITIVE_PATTERNS:
        if re.search(pattern, tweet_lower, re.IGNORECASE):
            return True
    
    # Check if it's just a location name
    if is_location_only(tweet):
        return True
    
    # Check if disaster keywords exist without proper context
    has_disaster_keywords = False
    for keywords in disaster_keywords.values():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', tweet_lower):
                has_disaster_keywords = True
                break
        if has_disaster_keywords:
            break
    
    # If has disaster keywords but no context, it's likely a false positive
    if has_disaster_keywords and not has_disaster_context(tweet):
        # Exception for very short tweets that are clearly disaster reports
        if len(tweet.split()) <= 3:
            return True
    
    return False

def extract_locations(tweet, language):
    """Enhanced location extraction using spaCy NER, TextBlob, and pattern matching."""
    locations = set()
    
    # Use spaCy for enhanced NER if available
    if nlp:
        try:
            doc = nlp(tweet)
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entities and locations
                    # Check if it matches known locations
                    for location in INDIAN_LOCATIONS:
                        if location.lower() in ent.text.lower():
                            locations.add(location)
                    
                    # Check aliases
                    for alias, location in city_aliases.items():
                        if alias.lower() in ent.text.lower():
                            locations.add(location)
        except Exception as e:
            logging.error(f"spaCy NER error: {e}")
    else:
        # Fallback NER using simple pattern matching
        logging.info("Using fallback location extraction (spaCy not available)")
    
    # Use TextBlob for additional NER (always available)
    try:
        blob = TextBlob(tweet)
        for noun_phrase in blob.noun_phrases:
            # Check if noun phrase matches known locations
            for location in INDIAN_LOCATIONS:
                if location.lower() in noun_phrase.lower():
                    locations.add(location)
    except Exception as e:
        logging.error(f"TextBlob NER error: {e}")
    
    # Check for location aliases with word boundaries
    tweet_lower = tweet.lower()
    for alias, location in city_aliases.items():
        if re.search(r'\b' + re.escape(alias.lower()) + r'\b', tweet_lower):
            locations.add(location)
    
    # Pattern matching for Indian locations with enhanced context
    for location in INDIAN_LOCATIONS:
        pattern = r'\b' + re.escape(location.lower()) + r'\b'
        if re.search(pattern, tweet_lower, re.IGNORECASE):
            locations.add(location)
    
    # Enhanced location detection for Indian states and cities
    indian_location_patterns = [
        r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # "in Delhi", "in West Bengal"
        r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # "at Mumbai", "at Chennai"
        r'\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'  # "from Bangalore"
    ]
    
    for pattern in indian_location_patterns:
        matches = re.finditer(pattern, tweet, re.IGNORECASE)
        for match in matches:
            potential_location = match.group(1)
            for location in INDIAN_LOCATIONS:
                if location.lower() == potential_location.lower():
                    locations.add(location)
    
    return list(locations)

def predict_tweet(tweet):
    """Enhanced prediction pipeline with improved contextual logic."""
    try:
        # Load models with version warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            
            with open('lr_model.pkl', 'rb') as f:
                lr_model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                st = pickle.load(f)
                
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}")
        return {'error': f"Model file not found: {e}"}
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return {'error': f"Critical error occurred while loading models."}
    
    # Validate input
    if not tweet or not isinstance(tweet, str) or len(tweet.strip()) < 3:
        return {'error': 'Invalid or too short tweet provided.'}
    
    # Analysis Pipeline
    cleaned_tweet = clean_tweet(tweet)
    if cleaned_tweet == 'Empty':
        return {'error': 'Tweet became empty after cleaning.'}
    
    print(f"Debug - Original tweet: {tweet}")
    print(f"Debug - Cleaned tweet: {cleaned_tweet}")
    
    # Enhanced false positive detection
    if is_false_positive(cleaned_tweet):
        print("Debug - Detected as false positive")
        return {
            'tweet': tweet,
            'is_disaster': 0,
            'confidence': 0.85,
            'location': 'Unknown',
            'category': 'Not Disaster',
            'sentiment': 'Neutral',
        }
    
    language = detect_language(cleaned_tweet)
    print(f"Debug - Detected language: {language}")
    translated_tweet = translate_tweet(cleaned_tweet, language)
    print(f"Debug - Translated tweet: {translated_tweet}")
    
    print(f"Debug - Has disaster context: {has_disaster_context(translated_tweet)}")
    
    # Check for disaster context but don't immediately reject
    has_context = has_disaster_context(translated_tweet)
    print(f"Debug - Has disaster context: {has_context}")
    
    # Only reject if it's clearly not disaster-related AND has no keywords
    if not has_context and not any(keyword.lower() in translated_tweet.lower() 
                                  for keywords in disaster_keywords.values() 
                                  for keyword in keywords):
        print("Debug - No disaster context or keywords detected")
        return {
            'tweet': tweet,
            'is_disaster': 0,
            'confidence': 0.75,
            'location': 'Unknown',
            'category': 'Not Disaster',
            'sentiment': 'Neutral',
        }
    
    # Vectorize and scale the tweet
    try:
        tweet_vectorized = vectorizer.transform([translated_tweet])
        tweet_scaled = st.transform(tweet_vectorized.toarray())
    except Exception as e:
        logging.error(f"Vectorization/scaling error: {e}")
        return {'error': 'Failed to process text for prediction.'}
    
    # Predict disaster probability with correct class handling
    try:
        # Get probabilities for both classes
        prediction_proba = lr_model.predict_proba(tweet_scaled)[0]
        
        # Check the class mapping to determine which index corresponds to disaster
        classes = lr_model.classes_
        print(f"Debug - Classes: {classes}")
        print(f"Debug - Probabilities: {prediction_proba}")
        
        # Find the index of the disaster class (1 = disaster, 0 = not disaster)
        if len(classes) == 2:
            # Binary classification: find index of class 1 (disaster)
            disaster_idx = list(classes).index(1) if 1 in classes else 1
        else:
            # Fallback to assuming second class is disaster
            disaster_idx = 1 if len(prediction_proba) > 1 else 0
        
        # Get disaster probability correctly with bounds checking
        if len(prediction_proba) > disaster_idx:
            disaster_prob = prediction_proba[disaster_idx]
        else:
            # Fallback: use the highest probability
            disaster_prob = max(prediction_proba)
        
        # Optimized dynamic threshold adjustment for better disaster detection
        context_strength = sum(1 for indicator in DISASTER_CONTEXT_INDICATORS 
                             if indicator.lower() in translated_tweet.lower())
        
        # Check for disaster keywords presence with better matching
        has_strong_keywords = False
        keyword_count = 0
        for keywords in disaster_keywords.values():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', translated_tweet.lower()):
                    has_strong_keywords = True
                    keyword_count += 1
        
        if context_strength >= 3 and keyword_count >= 2:
            adjusted_threshold = 0.25  # Strong context + multiple keywords
        elif context_strength >= 2 and has_strong_keywords:
            adjusted_threshold = 0.3   # Good context + keywords
        elif keyword_count >= 2:
            adjusted_threshold = 0.35  # Multiple keywords
        elif has_strong_keywords:
            adjusted_threshold = 0.4   # Single keyword - more sensitive
        elif context_strength >= 2:
            adjusted_threshold = 0.5   # Context without keywords
        elif context_strength >= 1:
            adjusted_threshold = 0.55  # Some context
        else:
            adjusted_threshold = 0.6   # Weak signals
        
        is_disaster = 1 if disaster_prob > adjusted_threshold else 0
        
        # Fix confidence calculation - should reflect actual prediction confidence
        if is_disaster:
            confidence = float(round(disaster_prob, 2))
        else:
            confidence = float(round(1 - disaster_prob, 2))
        
        print(f"Debug - Disaster probability: {disaster_prob}, Threshold: {adjusted_threshold}, Is disaster: {is_disaster}")
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        is_disaster = 0
        confidence = 0.0
    
    # Enhanced disaster category classification
    category = 'Unknown Category'
    category_confidence = 0.0
    
    if is_disaster:
        category_scores = {}
        tweet_lower = translated_tweet.lower()
        
        # Score each category based on keyword matches
        for cat, keywords in disaster_keywords.items():
            score = 0
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', tweet_lower):
                    # Weight keywords differently based on specificity
                    if len(keyword.split()) > 1:  # Multi-word keywords are more specific
                        score += 2
                    else:
                        score += 1
            
            if score > 0:
                category_scores[cat] = score
        
        # Select category with highest score
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            category = best_category[0].replace("_", " ").title()
            category_confidence = min(best_category[1] / 3.0, 1.0)  # Normalize confidence
        else:
            category = 'General Disaster'
            category_confidence = 0.5
    else:
        category = 'Not Disaster'
        category_confidence = confidence
    
    # Enhanced sentiment analysis with emotional context
    sentiment_scores = sia.polarity_scores(translated_tweet)
    compound_score = sentiment_scores['compound']
    
    # Enhanced sentiment classification with disaster context
    if is_disaster:
        # For disaster tweets, consider urgency and fear indicators
        urgency_words = ['help', 'urgent', 'emergency', 'rescue', 'trapped', 'danger']
        fear_words = ['scared', 'terrified', 'panic', 'afraid', 'worried', 'anxious']
        
        has_urgency = any(word in translated_tweet.lower() for word in urgency_words)
        has_fear = any(word in translated_tweet.lower() for word in fear_words)
        
        if has_urgency or has_fear:
            sentiment = 'Urgent/Fearful'
        elif compound_score <= -0.3:
            sentiment = 'Highly Negative'
        elif compound_score <= -0.05:
            sentiment = 'Negative'
        elif compound_score >= 0.05:
            sentiment = 'Concerned/Informative'
        else:
            sentiment = 'Neutral'
    else:
        # Standard sentiment for non-disaster tweets
        if compound_score >= 0.05:
            sentiment = 'Positive'
        elif compound_score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
    
    # Extract locations with enhanced detection
    locations = extract_locations(translated_tweet, 'en')
    
    return {
        'tweet': tweet,
        'is_disaster': is_disaster,
        'confidence': confidence,
        'location': locations[0] if locations else 'Unknown',
        'all_locations': locations,
        'category': category,
        'category_confidence': category_confidence,
        'sentiment': sentiment,
        'sentiment_score': compound_score,
        'language_detected': language,
        'translated_text': translated_tweet if language != 'en' and language != 'unknown' else None,
        'context_strength': context_strength,
        'has_keywords': has_strong_keywords,
        'keyword_count': keyword_count if 'keyword_count' in locals() else 0,
        'threshold_used': adjusted_threshold if 'adjusted_threshold' in locals() else base_threshold
    }

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a single tweet for disaster content."""
    try:
        data = request.get_json()
        if not data or 'tweet' not in data:
            return jsonify({'error': 'No tweet provided.'}), 400
        
        tweet = data['tweet']
        if not tweet or len(tweet.strip()) < 3:
            return jsonify({'error': 'Tweet is too short or empty.'}), 400
        
        result = predict_tweet(tweet)
        if 'error' in result:
            return jsonify(result), 500
            
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Unhandled error in /analyze: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during analysis.'}), 500

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Process uploaded file containing multiple tweets."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    
    # Securely save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                tweet = line.strip()
                if tweet and len(tweet) >= 3:
                    try:
                        res = predict_tweet(tweet)
                        if res and 'error' not in res:
                            res['line_number'] = line_num
                            results.append(res)
                    except Exception as e:
                        logging.error(f"Error processing line {line_num}: {e}")
                        continue
        
        # Clean up uploaded file
        os.remove(filepath)
        
        if not results:
            return jsonify({'error': 'No valid tweets to process in file.'}), 400
        
        # Create CSV response
        output = io.StringIO()
        fieldnames = ['line_number', 'tweet', 'is_disaster', 'confidence', 
                     'location', 'category', 'sentiment']
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=analysis_results.csv"
        response.headers["Content-type"] = "text/csv"
        return response
        
    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}", exc_info=True)
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'File processing failed.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3001)
