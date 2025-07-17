# 🧠 Brain Stroke Prediction Web App

This is a Flask-based web application that predicts the risk of brain stroke based on medical and personal information. It uses a **Logistic Regression** model trained on the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

---

## 🌐 Demo

<img width="1262" height="720" alt="image" src="https://github.com/user-attachments/assets/7cfa4354-dedd-4af8-be21-a182fc435ecd" />


---

## 📁 Folder Structure

```

Brain\_Stroke\_Prediction\_Project/
│
├── app.py                     # Flask application
├── train\_model.py             # Script to train Logistic Regression model
├── model/
│   └── stroke\_model.pkl       # Trained model (auto-generated after training)
├── stroke\_data.csv            # Dataset (download from Kaggle)
├── patient\_predictions.csv    # Logs saved predictions (auto-generated)
├── templates/
│   ├── index.html             # Frontend input form
│   └── result.html            # Result display page
├── static/
│   └── css/
│       └── style.css          # Beautiful CSS styling
└── requirements.txt           # Required Python libraries

````

---

## 🔍 Features

- 🖥️ Clean and responsive frontend (HTML/CSS)
- 🧠 Machine Learning backend using Logistic Regression
- ✅ Real-time predictions with result display
- 📊 Saves user input and predictions to CSV file
- 🧾 Form collects additional personal details (Name, Address, etc.)

---

## 🚀 How to Run

### 1. Clone the Repo or Download ZIP

```bash
git clone https://github.com/your-username/brain-stroke-prediction.git
cd brain-stroke-prediction
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
Rename it as: `stroke_data.csv` and place it in the project root folder.

### 4. Train the Model

```bash
python train_model.py
```

This will create `model/stroke_model.pkl`.

### 5. Run the Web App

```bash
python app.py
```

Open in browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📌 Input Fields in the Form

* Patient Name
* Father's Name
* Occupation
* Full Address
* Gender
* Age
* Hypertension
* Heart Disease
* Ever Married
* Work Type
* Residence Type
* Avg Glucose Level
* BMI
* Smoking Status

---

## 📁 Output

* Displays: ✅ “Low Risk” or ❗ “High Risk”
* Saves every patient's record with prediction in: `patient_predictions.csv`

---

## 🧠 Model Used

* **Algorithm**: Logistic Regression
* **Libraries**: scikit-learn, pandas, numpy

---

## 🛠 Technologies Used

* Python 🐍
* Flask 🔥
* HTML + CSS 🌐
* scikit-learn 🤖
* Pandas & NumPy 📊

---

## 📜 License

This project is for educational and academic purposes.
You are free to use, modify, and extend it for learning.

---

## ✨ Screenshots

<!-- Add screenshots in the `static/` folder and display them here -->

<img width="1262" height="720" alt="image" src="https://github.com/user-attachments/assets/292a8397-5f48-41bc-a5fe-029fedc715b8" />




---

## 📞 Contact

Created by Khushman — feel free to reach out!


