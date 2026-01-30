import json 
import os 
 
# Create notebooks directory 
os.makedirs('notebooks', exist_ok=True) 
 
# NOTEBOOK 1 - STRAWBERRY 
notebook1 = { 
    "cells": [ 
        { 
            "cell_type": "markdown", 
            "source": ["# African Agriculture Revolution\\n", "## Notebook 1: Strawberry Analysis"] 
        }, 
        { 
            "cell_type": "code", 
            "source": ["import pandas as pd\\n", "df = pd.read_csv('data/raw/strawberry_farmers_morogoro.csv')\\n", "print(f'Loaded {len(df)} strawberry farmers')"] 
        } 
    ], 
    "metadata": {}, 
    "nbformat": 4, 
    "nbformat_minor": 0 
} 
 
# NOTEBOOK 2 - ARUSHA 
notebook2 = { 
    "cells": [ 
        { 
            "cell_type": "markdown", 
            "source": ["# African Agriculture Revolution\\n", "## Notebook 2: Arusha Analysis"] 
        }, 
        { 
            "cell_type": "code", 
            "source": ["import pandas as pd\\n", "df = pd.read_csv('data/raw/both_crops_farmers_arusha.csv')\\n", "print(f'Loaded {len(df)} Arusha farmers')"] 
        } 
    ], 
    "metadata": {}, 
    "nbformat": 4, 
    "nbformat_minor": 0 
} 
 
# NOTEBOOK 3 - INTEGRATION 
notebook3 = { 
    "cells": [ 
        { 
            "cell_type": "markdown", 
            "source": ["# African Agriculture Revolution\\n", "## Notebook 3: Integrated Analysis"] 
        }, 
        { 
            "cell_type": "code", 
            "source": ["import pandas as pd\\n", "s = pd.read_csv('data/raw/strawberry_farmers_morogoro.csv')\\n", "a = pd.read_csv('data/raw/both_crops_farmers_arusha.csv')\\n", "print(f'Total: {len(s) + len(a)} Tanzanian farmers')"] 
        } 
    ], 
    "metadata": {}, 
    "nbformat": 4, 
    "nbformat_minor": 0 
} 
 
# NOTEBOOK 4 - PRODUCTION 
notebook4 = { 
    "cells": [ 
        { 
            "cell_type": "markdown", 
            "source": ["# African Agriculture Revolution\\n", "## Notebook 4: Production Pipeline"] 
        }, 
        { 
            "cell_type": "code", 
            "source": ["print('Production pipeline ready!')\\n", "print('Run: python run_simple.py')"] 
        } 
    ], 
    "metadata": {}, 
    "nbformat": 4, 
    "nbformat_minor": 0 
} 
 
# Save all notebooks 
with open('notebooks/01_Strawberry_Analysis.ipynb', 'w') as f: 
    json.dump(notebook1, f, indent=2) 
 
with open('notebooks/02_Arusha_Analysis.ipynb', 'w') as f: 
    json.dump(notebook2, f, indent=2) 
 
with open('notebooks/03_Integration_Analysis.ipynb', 'w') as f: 
    json.dump(notebook3, f, indent=2) 
 
with open('notebooks/04_Production_Pipeline.ipynb', 'w') as f: 
    json.dump(notebook4, f, indent=2) 
 
print('? Created 4 notebooks in notebooks/ folder') 
