import os
import sys

def check_setup():
    """Check if all required files and folders exist"""
    print("🔍 Checking project setup...")
    
    issues = []
    
    # Check folders
    folders = ['templates', 'datasets', 'models']
    for folder in folders:
        if not os.path.exists(folder):
            issues.append(f"❌ Missing folder: {folder}/")
    
    # Check required files
    required_files = [
        'app.py',
        'generate_datasets.py',
        'model_enhanced.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"❌ Missing file: {file}")
    
    # Check templates
    templates = ['index.html', 'quiz.html', 'results.html', 'career_details.html', 'error.html']
    for template in templates:
        if not os.path.exists(f'templates/{template}'):
            issues.append(f"❌ Missing template: templates/{template}")
    
    # Check if datasets are generated
    required_datasets = ['personality.csv', 'work_values.csv', 'training_data_comprehensive.csv']
    for dataset in required_datasets:
        if not os.path.exists(f'datasets/{dataset}'):
            issues.append(f"⚠️  Missing dataset: datasets/{dataset} (run generate_datasets.py)")
    
    # Check if model is trained
    model_files = ['career_rf_model.pkl', 'feature_scaler.pkl', 'sklearn_encoders.pkl']
    for model_file in model_files:
        if not os.path.exists(f'models/{model_file}'):
            issues.append(f"⚠️  Missing model: models/{model_file} (run model_enhanced.py)")
    
    if issues:
        print("\n⚠️  Setup Issues Found:\n")
        for issue in issues:
            print(f"  {issue}")
        print("\n📝 To fix these issues:")
        print("  1. Create missing folders: mkdir templates datasets models")
        print("  2. Run: python generate_datasets.py")
        print("  3. Run: python model_enhanced.py")
        print("  4. Make sure all HTML files are in templates/")
        return False
    else:
        print("✅ All checks passed!\n")
        return True

def main():
    """Main function to run the app"""
    print("="*60)
    print("🚀 CareerPath AI - Starting Application")
    print("="*60)
    
    # Check setup
    if not check_setup():
        print("\n❌ Please fix the issues above before running the app.")
        sys.exit(1)
    
    # Import and run Flask app
    try:
        from app import app
        
        print("="*60)
        print("✅ Server is running!")
        print("📍 Open your browser to: http://localhost:5000")
        print("🛑 Press CTRL+C to stop the server")
        print("="*60)
        print()
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"\n❌ Error importing app.py: {e}")
        print("⚠️  Make sure app.py exists and has no syntax errors")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()