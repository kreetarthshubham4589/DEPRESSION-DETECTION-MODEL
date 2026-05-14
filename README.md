# DEPRESSION-DETECTION-MODEL
## Overview
This project implements a depression text classification workflow using two modelling approaches:

- A classical machine learning pipeline based on TF-IDF features and a best-selected classifier between tuned Logistic Regression and SVM.
- A transformer-based pipeline using DistilBERT for higher-quality contextual text classification.

The repository also includes a Streamlit application for inference and an experiment notebook prepared specifically for Google Colab.

## Core Components

- [app.py](app.py): Streamlit application for local inference.
- [model.py](model.py): Script version of the end-to-end training pipeline.
- [model.ipynb](model.ipynb): Colab-only training notebook.
- [dataset.csv](dataset.csv): Dataset used for training.
- [requirements.txt](requirements.txt): Runtime dependencies.

## Recommended Workflow

The recommended usage flow is:

1. Train the models with [model.ipynb](model.ipynb) in Google Colab.
2. Obtain the generated artifacts:
   - `depression_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `distilbert_depression_model/`
3. Place those artifacts in the project root locally.
4. Run [app.py](app.py) with Streamlit.

## Prerequisites

### Local Application

- Python 3.13.x is recommended.
- A local terminal with permission to install Python packages.

### Google Colab Notebook

- A Google account for Colab.
- Access to Google Drive.
- The project dataset file: [dataset.csv](dataset.csv)

## Local Setup

Clone the repository and move into the project directory:

```powershell
git clone <repository-url>
cd project
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade `pip` and install the runtime dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Streamlit Application

### Important Requirement

The application requires the following trained artifacts to exist in the project root:

- `depression_model.pkl`
- `tfidf_vectorizer.pkl`
- `distilbert_depression_model/`

If these files are not available after cloning, generate them first by following the notebook instructions in the next section.

### Start the App

Run the application with Streamlit:

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit, which is usually:

```text
http://localhost:8501
```

An alternative Windows launcher is also included:

```powershell
.\run_app.cmd
```

## Running `model.ipynb` in Google Colab

### Important Note

[model.ipynb](model.ipynb) is intentionally configured for **Google Colab only**. It is not intended to run as a local Jupyter notebook.

### Step-by-Step Instructions

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload [model.ipynb](model.ipynb) to Colab.
3. Change the runtime type to `T4 GPU`.
4. Run the notebook from the beginning.
5. Approve Google Drive mounting when prompted.
6. Upload [dataset.csv](dataset.csv) if the notebook requests it.
7. Allow the notebook to complete all cells.

### What the Notebook Does

The notebook will:

- mount Google Drive
- create the working directory `/content/drive/MyDrive/depression_project`
- install the required notebook dependencies
- train the classical machine learning pipeline
- train the DistilBERT transformer pipeline
- evaluate all configured models
- save the trained artifacts back to Google Drive

### Output Files Produced in Colab

After training, the notebook saves the following to:

```text
/content/drive/MyDrive/depression_project
```

Generated artifacts:

- `depression_model.pkl`
- `tfidf_vectorizer.pkl`
- `distilbert_depression_model/`
- `transformer_results/`
- `logs/`

### After Notebook Completion

To run the Streamlit app locally, copy the following generated items from Google Drive into the project root on your machine:

- `depression_model.pkl`
- `tfidf_vectorizer.pkl`
- `distilbert_depression_model/`

Without these files, the application will fail during model loading.

## Optional Local Training Script

If required, [model.py](model.py) provides a script version of the same overall training workflow. However, the notebook is the recommended path because it is already prepared for Colab execution and GPU-assisted transformer training.

Run it locally only if you intentionally want a script-based training flow:

```powershell
python model.py
```

## Application Behavior

The Streamlit application provides:

- `ML (Fast)`: uses the saved classical machine learning model.
- `Transformer (Accurate)`: uses the saved DistilBERT model.

The app also includes confidence-aware guidance:

- above 50 percent depression confidence, it displays supportive precautions
- above 80 percent depression confidence, it displays precautions and India-based support contact numbers
