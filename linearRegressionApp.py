import streamlit as st
import pandas as pd
import pickle

# Load model
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

train_columns = ['total_bill', 'size', 'smoker_No', 'sex_Female', 'day_Fri', 'day_Sat', 'day_Sun', 'time_Dinner']

def main():
    st.title("Tip Predictio using Linear Regression")

    total_bill = st.number_input("Total Bill")
    size = st.number_input("Party Size", min_value=1, step=1)
    smoker = st.selectbox("Smoker", ['Yes', 'No'])
    sex = st.selectbox("Sex", ['Female', 'Male'])
    day = st.selectbox("Day", ['Thur', 'Fri', 'Sat', 'Sun'])
    time = st.selectbox("Time", ['Lunch', 'Dinner'])

    if st.button("Predict Tip"):
        input_dict = {
            'total_bill': total_bill,
            'size': size,
            'smoker': smoker,
            'sex': sex,
            'day': day,
            'time': time
        }
        input_df = pd.DataFrame([input_dict])

        # One-hot encode all categorical columns used in training
        input_dummies = pd.get_dummies(input_df[['smoker', 'sex', 'day', 'time']],drop_first=True).astype(int)

        # Drop original categorical columns and concat dummies with numeric columns
        input_numeric = input_df.drop(columns=['smoker', 'sex', 'day', 'time'])
        input_processed = pd.concat([input_numeric, input_dummies], axis=1)

        # Add missing columns from training data with 0
        for col in train_columns:
            if col not in input_processed.columns:
                input_processed[col] = 0

        # Reorder columns to match training data exactly
        input_processed = input_processed[train_columns]

        # Predict
        pred = model.predict(input_processed)[0]
        st.success(f"Predicted Tip: ${pred:.2f}")

if __name__ == "__main__":
    main()
