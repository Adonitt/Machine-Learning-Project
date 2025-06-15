# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(r"/ML Final Project/knn_foot_model_balanced.joblib")

st.set_page_config(
    page_title="Football Player Preferred Foot Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Player Preferred Foot Predictor")
st.write(
    """
This app predicts a football (soccer) player's preferred foot using a trained
**K-Nearest Neighbors** model. Adjust the attributes in the sidebar and click
**Predict** to see the prediction.
"""
)


@st.cache_resource(show_spinner=True)
def load_model(path: Path):
    if not path.exists():
        st.error(f"Model file not found at: {path}")
        return None

    try:
        model = joblib.load(path)
        return model
    except Exception as exc:
        st.exception(exc)
        return None


model = load_model(MODEL_PATH)
if model is None:
    st.stop()


def user_input_features() -> pd.DataFrame:
    sb = st.sidebar
    sb.header("Player Attributes")

    # Basic
    sb.subheader("Basic Information")
    age = sb.slider("Age", 16, 45, 25)

    # Ratings
    sb.subheader("Ratings")
    overall = sb.slider("Overall Rating", 40, 99, 70)
    potential = sb.slider("Potential", 40, 99, 80)
    special = sb.slider("Special (0-3000)", 0, 3000, 1500)
    international_rep = sb.slider("International Reputation (1-5)", 1, 5, 2)

    # Skills
    sb.subheader("Skills")
    weak_foot = sb.slider("Weak Foot (1-5)", 1, 5, 3)
    skill_moves = sb.slider("Skill Moves (1-5)", 1, 5, 3)
    work_rate = sb.selectbox(
        "Work Rate",
        [
            "High/High", "High/Medium", "High/Low",
            "Medium/High", "Medium/Medium", "Medium/Low",
            "Low/High", "Low/Medium", "Low/Low",
        ],
    )

    # Position
    sb.subheader("Position")
    position = sb.selectbox(
        "Position",
        [
            "ST", "CF", "LW", "RW", "LM", "CM", "RM", "CDM", "CAM",
            "LWB", "RWB", "LB", "CB", "RB", "GK",
        ],
    )

    # Physical
    sb.subheader("Physical Attributes")
    height = sb.slider("Height (cm)", 150, 210, 180)
    weight = sb.slider("Weight (kg)", 50, 120, 75)
    body_type = sb.selectbox(
        "Body Type",
        ["Lean", "Normal", "Stocky", "Unique"]
    )

    # Misc
    sb.subheader("Additional Information")
    nationality = sb.text_input("Nationality", "Argentina")
    club = sb.text_input("Club", "FC Barcelona")

    data = {
        "Age": age,
        "Overall": overall,
        "Potential": potential,
        "Special": special,
        "International Reputation": international_rep,
        "Weak Foot": weak_foot,
        "Skill Moves": skill_moves,
        "Work Rate": work_rate,
        "Body Type": body_type,
        "Position": position,
        "Height": f"{height}cm",
        "Weight": f"{weight}kg",
        "Nationality": nationality,
        "Club": club,
    }

    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

st.subheader("Entered Attributes")
st.dataframe(input_df, use_container_width=True)


def preprocess_input(df_in: pd.DataFrame, fitted_model) -> pd.DataFrame:
    """Preprocess the input data to match the model's training format."""
    df = df_in.copy()

    # Convert height and weight to numeric if they're strings
    if isinstance(df['Height'].iloc[0], str) and 'cm' in df['Height'].iloc[0]:
        df['Height'] = df['Height'].str.replace('cm', '').astype(float)

    if isinstance(df['Weight'].iloc[0], str) and 'kg' in df['Weight'].iloc[0]:
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

    # Define categorical columns for one-hot encoding
    categorical_cols = ['Work Rate', 'Body Type', 'Position', 'Nationality', 'Club']

    # Ensure all categorical columns are strings
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Get expected columns from the model
    expected_cols = list(getattr(fitted_model, "feature_names_in_", []))

    if not expected_cols:
        return df_encoded

    # Add missing columns with 0
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns to match training data
    df_encoded = df_encoded[expected_cols]

    return df_encoded


st.subheader("Prediction")

if st.button("Predict"):
    try:
        processed = preprocess_input(input_df, model)
        prediction = model.predict(processed)[0]
        proba = model.predict_proba(processed)[0]

        # Get the probability for the predicted class
        confidence = max(proba) * 100

        # Map prediction to Left/Right
        foot_mapping = {0: "Left", 1: "Right"}
        predicted_foot = foot_mapping.get(prediction, "Unknown")

        # Display result
        st.success(f"Predicted Preferred Foot: **{predicted_foot}** (Confidence: {confidence:.1f}%)")

        # Add some fun facts
        if predicted_foot == "Left":
            st.info("""
            **Did you know?** Left-footed players are quite rare in football, making up only about 25% of professional players. 
            Many top players like Messi and Maradona are left-footed!
            """)
        else:
            st.info("""
            **Did you know?** About 75% of footballers are right-footed. 
            While more common, right-footed players dominate many positions on the field.
            """)

    except Exception as exc:
        st.error("An error occurred during prediction:")
        st.exception(exc)
        st.write("Please ensure that all inputs are valid and the model was trained with matching feature names.")

st.markdown("---")
st.markdown("""
### How to use this app:
1. Adjust the player attributes in the sidebar
2. Click the **Predict** button to see the predicted preferred foot
3. The model will show the prediction along with the confidence level

### About the model:
- Trained using K-Nearest Neighbors algorithm
- Predicts whether a player is left or right-footed based on their attributes
- The model considers various factors including skills, physical attributes, and position
""")