"""
🔧 GOOGLE COLAB SETUP SCRIPT
One-click setup for African Agriculture Revolution
"""

import os
import subprocess
import sys

def setup_colab_environment():
    """Setup complete Colab environment for African agriculture analysis"""
    
    print("="*80)
    print("🌍 SETTING UP AFRICAN AGRICULTURE REVOLUTION ON GOOGLE COLAB")
    print("="*80)
    
    # 1. Install packages
    print("\n📦 STEP 1: INSTALLING REVOLUTIONARY PACKAGES")
    packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'scikit-learn',
        'xgboost',
        'streamlit',
        'pyyaml',
        'openpyxl',
        'networkx'
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    print("✅ All packages installed!")
    
    # 2. Mount Google Drive
    print("\n💾 STEP 2: MOUNTING GOOGLE DRIVE")
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted at /content/drive")
    
    # 3. Clone GitHub repository
    print("\n🔗 STEP 3: CLONING GITHUB REPOSITORY")
    
    github_url = "https://github.com/ErwinNanyaro/African-Agriculture-Revolution.git"
    
    if os.path.exists("African-Agriculture-Revolution"):
        print("✅ Repository already exists")
    else:
        subprocess.run(["git", "clone", github_url], check=True)
        print("✅ Repository cloned successfully")
    
    # 4. Navigate to project
    os.chdir("African-Agriculture-Revolution")
    print("📁 Project directory: " + os.getcwd())
    
    # 5. Create data directory structure
    print("\n📁 STEP 4: CREATING DATA DIRECTORY STRUCTURE")
    directories = [
        'data/raw',
        'data/processed',
        'results/visualizations',
        'results/models',
        'results/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    # 6. Create configuration file
    print("\n⚙️  STEP 5: CREATING CONFIGURATION")
    
    config_content = """
# African Agriculture Revolution Configuration
data_paths:
  strawberry: 'data/raw/strawberry_farmers_Morogoro.csv'
  arusha: 'data/raw/both_crops_farmers_Arusha.csv'

analysis:
  target_column: 'high_yield'
  problem_type: 'classification'
  
monetization:
  base_price_ngo: 2000
  base_price_business: 500
  base_price_government: 5000
"""
    
    config_path = "config/paths.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Configuration created: {config_path}")
    
    # 7. Create sample notebook
    print("\n📓 STEP 6: CREATING SAMPLE NOTEBOOK")
    
    notebook_content = """# African Agriculture Revolution - Starter Notebook

## 1. Import Revolutionary Modules
from src.data_loader import AfricanAgricultureDataLoader
from src.data_cleaner import AfricanAgricultureDataCleaner
from src.social_capital_calculator import SocialCapitalCalculator

## 2. Load Your Data
loader = AfricanAgricultureDataLoader()
data = loader.load_strawberry_data()

## 3. Run Complete Analysis
cleaner = AfricanAgricultureDataCleaner()
cleaned_data, _ = cleaner.clean_dataset(data, 'strawberry')

calculator = SocialCapitalCalculator()
results = calculator.calculate_comprehensive_index(cleaned_data)

print(f"🎉 Analysis complete! Processed {len(results)} farmers.")
"""
    
    notebook_path = "notebooks/00_Starter_Notebook.ipynb"
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    print(f"✅ Starter notebook created: {notebook_path}")
    
    # 8. List project structure
    print("\n📁 STEP 7: PROJECT STRUCTURE")
    subprocess.run(["find", ".", "-type", "f", "-name", "*.py"], capture_output=True)
    
    print("\n" + "="*80)
    print("✅ REVOLUTIONARY ENVIRONMENT SETUP COMPLETE!")
    print("="*80)
    print("\n🎯 NEXT STEPS:")
    print("   1. Upload your data files to data/raw/")
    print("   2. Run the starter notebook")
    print("   3. Check out the example analyses")
    print("   4. Start building your portfolio!")
    
    return True

if __name__ == "__main__":
    setup_colab_environment()