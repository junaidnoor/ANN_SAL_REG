# 💼 Salary Prediction Web App (ML + Streamlit)

This is a **Machine Learning-based Salary Prediction Web App** built using:

- 🧠 **Keras** (for model training)
- 🧪 **Scikit-learn** (for preprocessing)
- 📊 **Pandas** and **NumPy** (for data handling)
- 🌐 **Streamlit** (for web interface)

The app predicts the **estimated annual salary** based on the following inputs:

- 🎓 Education Level
- 💼 Job Role
- 📆 Years of Experience

---

## 🚀 Features

- Easy-to-use Streamlit interface
- Uses a trained deep learning regression model (`.h5`)
- One-hot encoding for categorical variables
- StandardScaler for input normalization
- Real-time salary predictions

---

## 🧪 How to Run Locally

### 0.1 Application Url
https://annsalreg-gjsuaygxbrw2vvxaxrzma7.streamlit.app/

### 1. Clone the repository
```bash
git clone https://github.com/junaidnoor/ANN_SAL_REG.git
cd ANN_SAL_REG

2. Install dependencies
Make sure Python 3.11+ is installed.

pip install -r requirements.txt
3. Run the Streamlit app

streamlit run salary_app.py

📁 Project Files
File	Description
salary_app.py	Main application script
sal_reg_mdel.h5	Trained deep learning model
Sal_Reg_scaler.pkl	StandardScaler for input normalization
onehot_encoder_education.pkl	OneHotEncoder for education levels
onehot_encoder_job.pkl	OneHotEncoder for job roles
expected_columns.pkl	List of expected model input features

📌 Author
Made with ❤️ by **Junaid Noor**  
📧 Email: junaid.noor30@gmail.com

If you found this useful, please ⭐ the repository!
