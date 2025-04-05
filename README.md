# 🏠 House Price Prediction (Beginner ML Project)

This is a beginner-friendly machine learning project to predict house prices using the California Housing dataset. The project walks through the typical ML pipeline — from data preprocessing to model training and evaluation.

## 📊 Dataset
We use the **California Housing dataset**, a well-known dataset in the machine learning community that includes features such as:
- Median income
- House age
- Average number of rooms
- Latitude/Longitude

## 🛠 Steps Followed
1. **Load dataset** from `sklearn.datasets`
2. **Preprocess data**:
   - Feature scaling using `StandardScaler`
   - Train/test split
3. **Model training** using:
   - `LinearRegression`
4. **Model evaluation** using:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
5. **Visualization** of predicted vs actual house prices using `matplotlib` and `seaborn`

## ⚙️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python main.py
