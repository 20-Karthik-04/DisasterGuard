# 🚨 DisasterGuard - AI-Powered Real-Time Disaster Detection System

[![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://python.org)
[![Node.js](https://img.shields.io/badge/Node.js-16+-green.svg)](https://nodejs.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-red.svg)](https://flask.palletsprojects.com)
[![Deployment](https://img.shields.io/badge/Deployed-Render-purple.svg)](https://render.com)

DisasterGuard is an intelligent disaster detection and analysis system that uses machine learning and natural language processing to identify and classify disaster-related content from social media posts in real-time. The system provides comprehensive analysis including disaster classification, location extraction, sentiment analysis, and multi-language support.

## 🎯 Problem Statement

In emergency situations, social media becomes a critical source of real-time information about disasters. However, the massive volume of posts makes it impossible to manually monitor and identify genuine disaster reports. Traditional keyword-based systems suffer from:

- **High false positive rates** (casual use of disaster terms)
- **Language barriers** in multilingual regions like India
- **Inability to understand context** and urgency
- **Poor location extraction** from unstructured text
- **Lack of sentiment analysis** for emergency prioritization

## 🚀 Key Features

### 🧠 **Intelligent Text Analysis**
- **Advanced Preprocessing**: URL removal, hashtag processing, mention handling
- **Context Recognition**: Distinguishes real disasters from metaphorical usage
- **Dynamic Thresholds**: Adaptive confidence based on context strength (0.25-0.6 range)

### 🌍 **Multi-Language Processing**
- **Language Detection**: Automatic identification of 12+ languages
- **Translation Pipeline**: Google Translate with TextBlob fallback
- **Regional Support**: Specialized handling for Hindi, Bengali, Tamil, Telugu, and other Indian languages

### 📍 **Location Intelligence**
- **Named Entity Recognition**: Advanced location extraction using spaCy and TextBlob
- **Geographic Database**: 150+ Indian cities and states with aliases
- **Pattern Matching**: Contextual location detection ("in Delhi", "at Mumbai", "from Bangalore")

### 🏷️ **Disaster Classification**
- **49 Disaster Categories**: From earthquakes to cyber attacks
- **Comprehensive Keywords**: 500+ disaster-related terms across all categories
- **Weighted Scoring**: Multi-word keywords receive higher priority scores

### 💭 **Sentiment & Urgency Analysis**
- **Disaster-Specific Sentiment**: Urgent/Fearful, Highly Negative, Concerned/Informative
- **Emergency Indicators**: Detection of help requests, panic signals, and urgency words
- **Context-Aware Classification**: Different sentiment handling for disaster vs. normal content

### 🛡️ **False Positive Filtering**
- **Pattern Recognition**: 10+ regex patterns for common false positives
- **Contextual Validation**: Ensures proper disaster context indicators
- **Metaphor Detection**: Filters casual usage ("movie disaster", "cooking disaster", "traffic disaster")

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │───▶│  Node.js API    │───▶│  Python ML      │
│   (HTML/CSS/JS) │    │   (Express)     │    │   Service       │
│     Port 3000   │    │    Port 3000    │    │   Port 3001     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────┐        ┌─────────────┐
                       │   Static    │        │   ML Models │
                       │   Assets    │        │   & Data    │
                       └─────────────┘        └─────────────┘
```

### **Component Breakdown**

**🖥️ Frontend Layer (Node.js + Express)**
- Modern responsive web interface with real-time analysis
- Batch processing capabilities and results visualization
- Multiple pages: Home, About, Feedback, Model Insights, Motivation

**🤖 ML Processing Service (Python + Flask)**
- Core machine learning and NLP processing
- Multi-language detection, translation, and analysis
- Location extraction, sentiment analysis, and classification

**📊 Data Layer**
- Pre-trained scikit-learn models (Logistic Regression, TF-IDF Vectorizer, StandardScaler)
- Comprehensive knowledge bases (disaster keywords, city aliases, location databases)
- JSON-based configuration files for easy updates

## 🔧 Technologies Used

### **Backend Technologies**
- **Python 3.11.9**: Core ML processing language
- **Flask 3.0.3**: Web framework for ML API
- **Node.js**: Frontend server and API gateway
- **Express.js**: Web application framework

### **Machine Learning & NLP**
- **scikit-learn 1.5.1**: Machine learning algorithms (Logistic Regression)
- **NLTK 3.9.1**: Natural language processing and sentiment analysis
- **TextBlob 0.17.1**: Text processing and language detection fallback
- **langdetect 1.0.9**: Primary language detection
- **deep-translator 1.11.4**: Multi-language translation (Python 3.13 compatible)

### **Data Processing**
- **NumPy 1.26.4**: Numerical computations
- **Joblib 1.4.2**: Model serialization and loading
- **Requests 2.32.3**: HTTP client for API calls

### **Web Technologies**
- **HTML5/CSS3**: Modern responsive frontend with CSS Grid and Flexbox
- **JavaScript (ES6+)**: Interactive user interface with async/await
- **Font Awesome 6.4.0**: Comprehensive icon library
- **Google Fonts**: Poppins and Roboto Mono typography

### **Deployment & DevOps**
- **Gunicorn 23.0.0**: WSGI server for production deployment
- **Flask-CORS 5.0.0**: Cross-origin resource sharing
- **Git**: Version control with comprehensive commit history
- **Render**: Cloud deployment platform with automatic deployments

## 🚀 Quick Start

### **Prerequisites**
- Python 3.11+ (recommended 3.11.9 for deployment compatibility)
- Node.js 16+ 
- Git

### **Installation**

1. **Clone the repository:**
```bash
git clone https://github.com/20-Karthik-04/DisasterGuard.git
cd DisasterGuard
```

2. **Set up Python environment:**
```bash
cd python_service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up Node.js environment:**
```bash
cd ../node_backend
npm install
```

### **Running the Application**

1. **Start the Python ML service:**
```bash
cd python_service
source venv/bin/activate  # Ensure virtual environment is activated
python app.py
# Service will start on http://localhost:3001
```

2. **Start the Node.js frontend (in a new terminal):**
```bash
cd node_backend
npm start
# Frontend will start on http://localhost:3000
```

3. **Open your browser and navigate to `http://localhost:3000`**

### **Testing the System**

Run the comprehensive test suite:
```bash
cd DisasterGuard
source venv/bin/activate
python test_predictions.py
```

## 📁 Project Structure

```
DisasterGuard/
├── 📁 python_service/              # ML Processing Service
│   ├── 🐍 app.py                   # Main Flask application (733 lines)
│   ├── 📋 requirements.txt         # Python dependencies
│   ├── 🤖 lr_model.pkl            # Trained Logistic Regression model
│   ├── 📊 vectorizer.pkl          # TF-IDF Vectorizer
│   ├── ⚖️ scaler.pkl              # StandardScaler for feature scaling
│   ├── 🏷️ disaster_keywords.json  # 49 disaster categories with keywords
│   ├── 🌍 city_aliases.json       # Global city aliases and abbreviations
│   ├── 📝 disaster_errors.log     # Error logging file
│   └── 🐍 runtime.txt             # Python version for deployment
├── 📁 node_backend/               # Frontend Service
│   ├── 🚀 app.js                  # Express server configuration
│   ├── 📁 routes/
│   │   └── 🛣️ predict.js          # API routing for predictions
│   ├── 📁 public/
│   │   ├── 📁 templates/          # HTML pages
│   │   │   ├── 🏠 index.html      # Main interface (733 lines)
│   │   │   ├── ℹ️ about-us.html   # About page
│   │   │   ├── 💭 feedback.html   # Feedback form
│   │   │   ├── 💡 motivation.html # Project motivation
│   │   │   └── 🔍 model-insight.html # Model details
│   │   └── 📁 static/             # CSS, JS, images
│   └── 📦 package.json            # Node.js dependencies
├── 🧪 test_predictions.py         # Comprehensive testing script (268 lines)
├── 📊 IMPROVEMENTS.md             # Detailed improvement documentation
├── 📖 README.md                   # This file
├── 🤖 rf_pipeline_model_bert_only.joblib # Alternative model file
├── 📝 train.txt                   # Training data sample
└── 🚫 .gitignore                  # Git ignore rules
```

## 🔄 Data Flow Pipeline

1. **Input Processing**: Tweet text received via REST API
2. **Preprocessing**: Text cleaning, URL removal, hashtag processing
3. **Language Detection**: Automatic language identification using langdetect
4. **Translation**: Convert non-English text to English using deep-translator
5. **Feature Extraction**: TF-IDF vectorization and standard scaling
6. **ML Prediction**: Logistic Regression model generates disaster probability
7. **Context Analysis**: Dynamic threshold adjustment based on context strength
8. **Location Extraction**: NER and pattern matching for geographic entities
9. **Sentiment Analysis**: VADER sentiment with disaster-specific categories
10. **Response Generation**: Comprehensive JSON result with all metadata

## 🌐 Deployment

### **Render Deployment**
The application is configured for deployment on Render cloud platform:

- **Python Service**: Automatically deploys from `python_service/` directory
- **Runtime**: Python 3.11.9 (specified in `runtime.txt`)
- **Dependencies**: Auto-installed from `requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`

### **Environment Configuration**
```bash
# Production environment variables
FLASK_ENV=production
NLTK_DATA=/app/nltk_data
PORT=3001  # Render assigns this automatically
```

## 🔮 Future Enhancements

### **Immediate Improvements**
1. **Enhanced ML Models**: Implement BERT-based transformers for better accuracy
2. **Real-time Streaming**: Integration with Twitter API for live monitoring
3. **Geographic Clustering**: Spatial analysis of disaster reports
4. **Image Analysis**: OCR and image classification for multimedia posts

### **Scalability Features**
1. **Caching Layer**: Redis for frequently accessed data and model caching
2. **Load Balancing**: Multiple service instances with container orchestration
3. **Database Integration**: PostgreSQL for historical analysis and user management
4. **Monitoring**: Application performance monitoring with alerts

### **Advanced Analytics**
1. **Temporal Analysis**: Time-series analysis for disaster progression tracking
2. **Credibility Scoring**: Source reliability and verification metrics
3. **Impact Assessment**: Severity scoring based on multiple factors
4. **Predictive Analytics**: Early warning systems based on social media trends

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper documentation
4. **Add tests** for new functionality
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Submit a Pull Request** with detailed description



---
