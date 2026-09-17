#!/usr/bin/env python3
"""
Script to execute all Jupyter notebooks in sequence.
"""

import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import json

def run_notebook(notebook_path, output_dir="notebook_outputs"):
    """Execute a Jupyter notebook and save results."""
    
    print(f"\n{'='*80}")
    print(f"Executing: {os.path.basename(notebook_path)}")
    print(f"{'='*80}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Configure the execute preprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        
        # Save the executed notebook
        output_path = os.path.join(output_dir, f"executed_{os.path.basename(notebook_path)}")
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"✓ Notebook executed successfully: {output_path}")
        
        # Also save as HTML for easy viewing
        html_exporter = HTMLExporter()
        (body, resources) = html_exporter.from_notebook_node(nb)
        
        html_path = os.path.join(output_dir, f"{os.path.basename(notebook_path).replace('.ipynb', '.html')}")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(body)
        
        print(f"✓ HTML export created: {html_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error executing notebook: {e}")
        
        # Save the notebook with error for debugging
        error_path = os.path.join(output_dir, f"error_{os.path.basename(notebook_path)}")
        with open(error_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"✗ Saved notebook with error: {error_path}")
        return False

def main():
    """Main function to execute all notebooks in sequence."""
    
    notebooks = [
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_preprocessing.ipynb", 
        "notebooks/03_feature_engineering.ipynb",
        "notebooks/04_ml_models.ipynb",
        "notebooks/05_dl_models.ipynb"
    ]
    
    print("Starting notebook execution pipeline...")
    print(f"Working directory: {os.getcwd()}")
    
    results = {}
    
    for notebook_path in notebooks:
        if os.path.exists(notebook_path):
            success = run_notebook(notebook_path)
            results[notebook_path] = success
        else:
            print(f"✗ Notebook not found: {notebook_path}")
            results[notebook_path] = False
    
    print(f"\n{'='*80}")
    print("Execution Summary:")
    print(f"{'='*80}")
    
    for notebook_path, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {os.path.basename(notebook_path)}")
    
    # Overall status
    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 All notebooks executed successfully!")
    else:
        print(f"\n⚠️ Some notebooks failed. Check the output above for details.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())