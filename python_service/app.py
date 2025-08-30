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
import csv

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

# Initialize sentiment analysis
try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    logging.error(f"Failed to load NLP tools: {e}")
    print(f"CRITICAL ERROR: Failed to load NLP tools. Error: {e}")
    raise

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
    """Simple language detection using TextBlob; defaults to English."""
    try:
        if len(tweet.strip()) < 3:
            return 'en'
        blob = TextBlob(tweet)
        detected = blob.detect_language()
        return detected if detected else 'en'
    except Exception as e:
        logging.warning(f"Language detection failed for tweet: '{tweet}'. Error: {e}")
        return 'en'

def translate_tweet(tweet, source_lang):
    """Translates a tweet to English using TextBlob if not already in English."""
    if source_lang == 'en' or not tweet or len(tweet.strip()) < 3:
        return tweet
    
    try:
        blob = TextBlob(tweet)
        translated = blob.translate(to='en')
        return str(translated) if translated else tweet
    except Exception as e:
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
    """Check if tweet has proper disaster context indicators."""
    tweet_lower = tweet.lower()
    
    # Count contextual indicators
    context_count = sum(1 for indicator in DISASTER_CONTEXT_INDICATORS 
                       if indicator.lower() in tweet_lower)
    
    # Look for disaster keywords with context
    disaster_with_context = False
    for category, keywords in disaster_keywords.items():
        for keyword in keywords:
            if keyword.lower() in tweet_lower:
                # Check if disaster word has proper context around it
                keyword_pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                match = re.search(keyword_pattern, tweet_lower)
                if match:
                    start, end = match.span()
                    # Get context before and after the keyword
                    before_text = tweet_lower[max(0, start-50):start]
                    after_text = tweet_lower[end:end+50]
                    
                    # Check for context indicators near the disaster word
                    context_near = any(indicator in before_text + after_text 
                                     for indicator in DISASTER_CONTEXT_INDICATORS)
                    
                    # Check for action verbs or impact words
                    action_words = ["hit", "struck", "caused", "killed", "injured", 
                                  "destroyed", "damaged", "occurred", "happened"]
                    has_action = any(word in before_text + after_text 
                                   for word in action_words)
                    
                    if context_near or has_action:
                        disaster_with_context = True
                        break
        
        if disaster_with_context:
            break
    
    return context_count >= 1 or disaster_with_context

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
        if any(keyword.lower() in tweet_lower for keyword in keywords):
            has_disaster_keywords = True
            break
    
    # If has disaster keywords but no context, it's likely a false positive
    if has_disaster_keywords and not has_disaster_context(tweet):
        # Exception for very short tweets that are clearly disaster reports
        if len(tweet.split()) <= 3:
            return True
    
    return False

def extract_locations(tweet, language):
    """Location extraction using TextBlob NER and pattern matching."""
    locations = set()
    
    # Use TextBlob for basic NER
    try:
        blob = TextBlob(tweet)
        for noun_phrase in blob.noun_phrases:
            # Check if noun phrase matches known locations
            for location in INDIAN_LOCATIONS:
                if location.lower() in noun_phrase.lower():
                    locations.add(location)
    except Exception as e:
        logging.error(f"TextBlob NER error: {e}")
    
    # Check for location aliases
    tweet_lower = tweet.lower()
    for alias, location in city_aliases.items():
        if re.search(r'\b' + re.escape(alias.lower()) + r'\b', tweet_lower):
            locations.add(location)
    
    # Pattern matching for Indian locations
    for location in INDIAN_LOCATIONS:
        if re.search(r'\b' + re.escape(location.lower()) + r'\b', tweet_lower, re.IGNORECASE):
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
    translated_tweet = translate_tweet(cleaned_tweet, language)
    
    print(f"Debug - Has disaster context: {has_disaster_context(translated_tweet)}")
    
    # Additional context check - if no disaster context, likely not a disaster
    if not has_disaster_context(translated_tweet):
        print("Debug - No disaster context detected")
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
        
        # Find the index of the disaster class
        if 1 in classes:
            disaster_idx = list(classes).index(1)
        elif 'disaster' in [str(c).lower() for c in classes]:
            disaster_idx = [str(c).lower() for c in classes].index('disaster')
        else:
            disaster_idx = 1 if len(prediction_proba) > 1 else 0
        
        # Get disaster probability correctly
        disaster_prob = prediction_proba[disaster_idx] if len(prediction_proba) > disaster_idx else prediction_proba[0]
        
        # Enhanced threshold logic with context consideration
        base_threshold = 0.5
        
        # Adjust threshold based on context strength
        context_strength = sum(1 for indicator in DISASTER_CONTEXT_INDICATORS 
                             if indicator.lower() in translated_tweet.lower())
        
        if context_strength >= 3:
            adjusted_threshold = 0.3  # Lower threshold for strong context
        elif context_strength >= 2:
            adjusted_threshold = 0.4  # Moderate adjustment
        else:
            adjusted_threshold = base_threshold
        
        is_disaster = 1 if disaster_prob > adjusted_threshold else 0
        confidence = float(round(max(disaster_prob, 1 - disaster_prob), 2))
        
        print(f"Debug - Disaster probability: {disaster_prob}, Threshold: {adjusted_threshold}, Is disaster: {is_disaster}")
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        is_disaster = 0
        confidence = 0.0
    
    # Determine disaster category
    category = 'Unknown Category'
    if is_disaster:
        for cat, keywords in disaster_keywords.items():
            if any(re.search(r'\b' + re.escape(word.lower()) + r'\b', 
                           translated_tweet.lower()) for word in keywords):
                category = cat.replace("_", " ").title()
                break
        if category == 'Unknown Category':
            category = 'General Disaster'
    else:
        category = 'Not Disaster'
    
    # Analyze sentiment
    sentiment_scores = sia.polarity_scores(translated_tweet)
    sentiment = ('Positive' if sentiment_scores['compound'] >= 0.05 
                else 'Negative' if sentiment_scores['compound'] <= -0.05 
                else 'Neutral')
    
    # Extract locations with enhanced detection
    locations = extract_locations(translated_tweet, 'en')
    
    return {
        'tweet': tweet,
        'is_disaster': is_disaster,
        'confidence': confidence,
        'location': locations[0] if locations else 'Unknown',
        'category': category,
        'sentiment': sentiment
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
    port = int(os.environ.get('PORT', 4000))
    app.run(debug=False, host='0.0.0.0', port=port)
