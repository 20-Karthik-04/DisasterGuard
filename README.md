# DisasterGuard - Real-time Disaster Tweet Analysis

A comprehensive AI-powered system for analyzing social media posts to detect disaster-related content in real-time. The application uses machine learning to classify text, extract locations, and provide sentiment analysis for emergency response coordination.

## 🚀 Features

- **Real-time Text Analysis**: Instant disaster detection from social media posts
- **Multi-language Support**: Automatic language detection and translation
- **Location Extraction**: Enhanced NER for Indian and international locations
- **Sentiment Analysis**: Emotional context understanding
- **Category Classification**: Disaster type identification (earthquake, flood, hurricane, etc.)
- **False Positive Filtering**: Advanced contextual analysis to reduce noise
- **Modern UI**: Dark/light mode toggle with responsive design
- **File Upload**: Batch processing of multiple texts via CSV

## 🏗️ Architecture

### Frontend (Node.js)
- **Express.js** server serving static HTML templates
- **Responsive UI** with modern CSS and JavaScript
- **Real-time communication** with Python backend

### Backend (Python Flask)
- **Machine Learning Pipeline** with scikit-learn
- **NLP Processing** using spaCy and NLTK
- **Multi-language Support** with Google Translate API
- **Enhanced Location Detection** for Indian geography

## 📋 Prerequisites

- **Node.js** (v16 or higher)
- **Python** (3.8 or higher)
- **pip** package manager

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd dt-copy
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Node.js Dependencies
```bash
cd node_backend
npm install
```

### 4. Download Required Models
```bash
# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
```

## 🚀 Running the Application

### Development Mode

1. **Start Python Backend** (Terminal 1):
```bash
cd python_service
python app.py
```
The Python service will run on `http://localhost:4000`

2. **Start Node.js Frontend** (Terminal 2):
```bash
cd node_backend
npm start
```
The frontend will be available at `http://localhost:3000`

### Production Mode

The application is configured for deployment on Render with automatic builds and environment setup.

## 🌐 Deployment on Render

### Backend Service (Python)
1. Create a new **Web Service** on Render
2. Connect your GitHub repository
3. Set **Root Directory**: `python_service`
4. **Build Command**: `pip install -r requirements.txt && python -m spacy download en_core_web_sm && python -m spacy download xx_ent_wiki_sm`
5. **Start Command**: `gunicorn app:app --host 0.0.0.0 --port $PORT`
6. Set environment variables if needed

### Frontend Service (Node.js)
1. Create a new **Web Service** on Render
2. Connect your GitHub repository
3. Set **Root Directory**: `node_backend`
4. **Build Command**: `npm install`
5. **Start Command**: `npm start`
6. Update the API endpoint in the frontend to point to your deployed Python service

## 📁 Project Structure

```
dt-copy/
├── python_service/           # Flask backend
│   ├── app.py               # Main Flask application
│   ├── requirements.txt     # Python dependencies
│   ├── *.pkl               # Pre-trained ML models
│   ├── *.json              # Configuration files
│   └── uploads/            # File upload directory
├── node_backend/            # Express frontend
│   ├── app.js              # Express server
│   ├── package.json        # Node.js dependencies
│   └── public/
│       ├── static/         # CSS, JS, images
│       └── templates/      # HTML templates
├── requirements.txt         # Global Python requirements
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 4000 for Python, 3000 for Node.js)
- `FLASK_ENV`: Set to `production` for deployment
- `NODE_ENV`: Set to `production` for deployment

### Model Files
The application requires these pre-trained models:
- `lr_model.pkl`: Logistic regression classifier
- `vectorizer.pkl`: Text vectorizer
- `scaler.pkl`: Feature scaler

## 🧪 API Endpoints

### POST /analyze
Analyze a single text for disaster content.

**Request:**
```json
{
  "tweet": "Earthquake hits Mumbai, buildings collapsed"
}
```

**Response:**
```json
{
  "tweet": "Earthquake hits Mumbai, buildings collapsed",
  "is_disaster": 1,
  "confidence": 0.95,
  "location": "Mumbai",
  "category": "Earthquake",
  "sentiment": "Negative"
}
```

### POST /upload_file
Process multiple texts from uploaded file.

**Request:** Multipart form with text file
**Response:** CSV file with analysis results

## 🎯 Usage Examples

### Single Text Analysis
1. Navigate to the main page
2. Enter text in the input field
3. Click "Analyze Text"
4. View detailed results including disaster classification, location, and sentiment

### Batch Processing
1. Prepare a text file with one text per line
2. Use the file upload feature
3. Download the CSV results with analysis for each text

## 🔍 Model Details

### Disaster Detection
- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Features**: Text preprocessing, keyword matching, contextual analysis
- **Accuracy**: Optimized for disaster-related content with false positive filtering

### Location Extraction
- **NER Models**: spaCy English and multilingual models
- **Enhanced Detection**: Custom Indian location database
- **Alias Support**: City and state abbreviations

### Sentiment Analysis
- **Tool**: NLTK VADER sentiment analyzer
- **Output**: Positive, Negative, or Neutral classification

## 🛡️ Security Features

- **Input Validation**: Comprehensive text sanitization
- **File Upload Security**: Secure filename handling
- **Error Handling**: Graceful error management with logging
- **CORS Configuration**: Proper cross-origin resource sharing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API endpoints

## 🔄 Version History

- **v1.0.0**: Initial release with basic disaster detection
- **v1.1.0**: Enhanced location detection and UI improvements
- **v1.2.0**: Production deployment configuration

---

**DisasterGuard** - Empowering emergency response through intelligent social media analysis.
