# Blood Cell Classification Using CNN

A Convolutional Neural Network (CNN) that classifies individual blood cells — cropped from microscopic blood-smear images — into **Red Blood Cells (RBC)** and **White Blood Cells (WBC)**. The trained model is served through a simple **Streamlit** web app that lets a user upload an image and view the predicted class.

**Live demo:** https://blood-cell-classification-cnn.streamlit.app

---

## Project Overview

- **Problem:** Automate blood cell classification to support faster, more consistent analysis of blood-smear images.
- **Approach:** Extract individual cell crops from an annotated blood-smear dataset using bounding boxes, then train a CNN to classify each crop as RBC or WBC.
- **Dataset:** Blood Cell Detection Dataset (annotated blood-smear images with bounding boxes and class labels).
- **Result:** ~99.7% test accuracy, with a full evaluation (classification report + confusion matrix) documented in the notebook.

## Repository Contents

| File / Folder | Description |
|---|---|
| `Blood Cell Classification - Rimsha Pervaiz.ipynb` | Full project notebook — data loading, preprocessing, augmentation, CNN training, and evaluation, with markdown explanations for each step. |
| `SRS - Blood Cell Classification - Rimsha Pervaiz.pdf` | Software Requirements Specification document. |
| `app.py` | Streamlit web app — loads the trained model and predicts on a user-uploaded image. |
| `requirements.txt` | Python dependencies needed to run the notebook and the app. |
| `runtime.txt` | Python runtime version for Streamlit Cloud deployment. |
| `assets/` | Supporting files used by the app (e.g. the saved model). |
| `.streamlit/` | Streamlit configuration. |

## How the Model Was Built

1. **Data preprocessing** — cell crops are extracted from annotated bounding boxes, resized to a fixed size, and pixel values normalized to [0, 1]; augmentation (rotation/flip/zoom) is applied to improve generalization given class imbalance between RBC and WBC samples.
2. **Model** — a Sequential CNN (convolution + pooling layers followed by dense layers) built with TensorFlow/Keras.
3. **Training** — early stopping and class weighting are used to handle the imbalance between RBC and WBC samples and reduce overfitting.
4. **Evaluation** — test accuracy/loss, a classification report (precision, recall, F1), and a confusion matrix.

## Running Locally

```bash
# Clone the repository
git clone https://github.com/rimiikhan2020-bot/blood-cell-classification-cnn.git
cd blood-cell-classification-cnn

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook "Blood Cell Classification - Rimsha Pervaiz.ipynb"

# Run the Streamlit app
streamlit run app.py
```

## Tech Stack

Python · TensorFlow / Keras · NumPy · Pandas · Matplotlib / Seaborn · Scikit-learn · Streamlit

## Author

**Rimsha Pervaiz**
Submitted as a CNN capstone project for the NAVTTC Artificial Intelligence (Machine Learning & Deep Learning) course.
