# 🎯 Multi-Modal Product Success Prediction
===

Production-ready machine learning system for predicting product success using both product metadata and visual features. The system combines tabular data with image embeddings to provide accurate success probability predictions.

---

## 🚀 Features
===

- **Multi-Modal Architecture**: Combines product metadata with visual features
- **Production API**: FastAPI with automatic Swagger documentation
- **MLOps Ready**: Model versioning, experiment tracking, monitoring
- **Real-time Predictions**: REST API for instant product analysis
- **Feature Importance**: Explainable AI with feature analysis

---

## 📊 Model Performance
===

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.75 |
| F1-Score | 0.62 |
| Accuracy | 0.65 |
| Feature Count | 50 |

---

## 🏗️ System Architecture
===
Data Input → Feature Engineering → Multi-Modal Model → API Service → Prediction
↓ ↓ ↓ ↓ ↓
Product Tabular + RandomForest FastAPI Success
Metadata Image Embeddings Model REST API Probability

```

## 📁 Project Structure
product-image-analysis/
├── 📁 models_improved/ # Trained models and feature definitions
│ ├── product_success_model.joblib
│ └── feature_names.txt
├── 📁 results_improved/ # Model metrics and analysis
│ ├── metrics.json
│ ├── feature_importance.csv
│ └── feature_importance.png
├── 📁 api/ # FastAPI application
│ └── app.py
├── app.py # Main application entry point
├── run_server.py # Server startup script
├── test_final.py # API testing suite
├── requirements.txt # Python dependencies
└── README.md # This file

```

## 🛠️ Installation
===

### Prerequisites
- Python 3.8+
- pip
  
---

### Setup
```
# Clone repository
git clone <your-repo-url>
cd product-image-analysis

# Create virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

🚀 Quick Start
===

1. Start the API Server
```
python run_server.py
2. Test the API (in another terminal)
```bash```
python test_final.py
3. Access the Web Interface
Main Dashboard: http://localhost:8000

API Documentation: http://localhost:8000/docs

Health Check: http://localhost:8000/health

```

---

📚 API Endpoints
===

🔍 Health Check
```
GET /health
Response:

```json```
{
  "status": "healthy",
  "model_loaded": true,
  "features_loaded": true,
  "details": {
    "model_type": "RandomForestClassifier",
    "model_features": 50,
    "feature_count": 50
  }
}
```
---

🤖 Model Information
===

```
GET /model/info
Returns model specifications and configuration.
```

---

📊 Model Metrics
===

```
GET /model/metrics
Returns performance metrics and evaluation results.
```
---

🎯 Make Prediction
===

```
POST /predict
Request:

```json```
{
  "product_data": {
    "id": 1,
    "gender": "Men",
    "masterCategory": "Apparel",
    "subCategory": "Topwear",
    "articleType": "Tshirts",
    "baseColour": "Black",
    "season": "Summer",
    "year": 2023,
    "usage": "Casual",
    "productDisplayName": "Test Product"
  },
  "image_url": ""
}
```
Response:

```
{
  "product_id": 1,
  "success_probability": 0.475,
  "prediction": 0,
  "confidence": 0.525,
  "message": "📈 Close to success. Minor optimizations could help."
}
```

---

📈 Feature Importance
===

```
GET /features/importance
Returns top 10 most important features for model predictions.
```

---

🧪 Testing
===

Run Complete Test Suite
```
python test_final.py

```
Manual Testing with curl
```
# Health check
curl http://localhost:8000/health
```
```
# Make prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "product_data": {
         "id": 1,
         "gender": "Men",
         "masterCategory": "Apparel",
         "subCategory": "Topwear", 
         "articleType": "Tshirts",
         "baseColour": "Black",
         "season": "Summer",
         "year": 2023,
         "usage": "Casual",
         "productDisplayName": "Test Product"
       },
       "image_url": ""
     }'
```

---

🔧 Technical Details
===

Model Architecture

* Algorithm: Random Forest Classifier

* Features: 50 total (22 tabular + 28 visual embeddings)

* Training: Balanced with SMOTE, cross-validated

---

Feature Engineering

* Tabular Features: One-hot encoded categorical variables

* Visual Features: ResNet50 embeddings (reduced dimensionality)

* Feature Selection: Top 50 most important features selected

---

Key Features

The model considers:

* Product categories and types

* Color and seasonal information

* Usage patterns and demographics

* Visual characteristics from product images

📊 Feature Importance
===

Top 5 most influential features:

* emb_25 - Visual embedding feature (7.7%)

* emb_18 - Visual embedding feature (6.8%)

* emb_12 - Visual embedding feature (6.0%)

* emb_5 - Visual embedding feature (5.1%)

* articleType_Shoes - Product type feature (3.0%)

---

🎯 Business Applications
===

E-commerce: Predict product performance before launch

Inventory Management: Optimize stock based on predicted success

Marketing: Target promotions for high-potential products

Product Development: Identify successful product characteristics

---

🔮 Prediction Interpretation
===

| Probability Range | Interpretation | Recommendation |
|--------|-------|-------|
| > 80% | 🎉 Excellent | confidence in success |
| 60-80% | ✅ High | Likely to perform well |
| 40-60% | 📊 Good| Fair	Needs minor optimizations |
| 20-40% | 📉 Below | Average	Significant improvements needed |
| < 20% | ⚠️ Low | Major changes recommended |

---

🛠️ Development
===

Adding New Features

Update feature engineering in training pipeline

Retrain model with new features

Update feature_names.txt with new feature names

Deploy updated model

---

Model Retraining

The model can be retrained with new data by running the training pipeline and updating the model files in models_improved/.

---

📝 License
===

MIT License - feel free to use this project for learning and development purposes.

---

🤝 Contributing
===

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

---

📞 Support
===

For questions or support, please contact the development team or open an issue in the repository.

---

**Built with ❤️ using FastAPI, Scikit-learn, and PyTorch**
