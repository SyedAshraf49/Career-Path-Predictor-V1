# Career Path Predictor

A machine learning-based web application that predicts suitable career paths based on user education, interests, skills, and personality traits.

## Features

- **Career Prediction**: ML model predicts top 5 career matches with confidence scores
- **Personality Quiz**: Interactive assessment of Big Five personality traits
- **Career Insights**: Detailed information about careers (salary, education, growth outlook)
- **Learning Paths**: Personalized roadmaps to transition to predicted careers
- **Multiple Languages**: Support for multiple education levels and specializations

## Tech Stack

- **Backend**: Flask, Python
- **ML**: Scikit-learn (Random Forest), joblib
- **Frontend**: HTML, CSS, JavaScript
- **Data**: CSV datasets for careers, skills, education paths

## Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## Project Structure

- `app.py` - Main Flask application
- `model.py` - ML model training script
- `datasets/` - Data files for careers, skills, education paths
- `templates/` - HTML templates for web interface
- `requirements.txt` - Python dependencies

## Environment Variables

Set `SECRET_KEY` environment variable for production:
```bash
export SECRET_KEY="your-secure-key-here"
```

## License

MIT
