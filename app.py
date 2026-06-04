import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Aerospace Voice Command AI",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Aerospace Voice Command Recognition System")
st.write("Machine Learning Based Aerospace Command Predictor")

data = {
    "command": [
        "take off",
        "land",
        "increase altitude",
        "decrease altitude",
        "activate autopilot",
        "deactivate autopilot",
        "check fuel status",
        "emergency stop",
        "increase speed",
        "decrease speed",
        "open cargo bay",
        "close cargo bay",
        "start engine",
        "stop engine",
        "check navigation",
        "activate radar",
        "deactivate radar",
        "emergency landing",
        "raise landing gear",
        "lower landing gear"
    ],

    "action": [
        "Aircraft Taking Off",
        "Aircraft Landing",
        "Increasing Altitude",
        "Decreasing Altitude",
        "Autopilot Activated",
        "Autopilot Deactivated",
        "Checking Fuel Status",
        "Emergency Protocol Initiated",
        "Increasing Speed",
        "Decreasing Speed",
        "Cargo Bay Opened",
        "Cargo Bay Closed",
        "Engine Started",
        "Engine Stopped",
        "Checking Navigation",
        "Radar Activated",
        "Radar Deactivated",
        "Emergency Landing Activated",
        "Landing Gear Raised",
        "Landing Gear Lowered"
    ]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["command"])
y = df["action"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.header("Model Information")
st.sidebar.metric("Accuracy", f"{accuracy*100:.2f}%")
st.sidebar.metric("Commands", len(df))
st.sidebar.metric("Algorithm", "Naive Bayes")

with st.expander("View Dataset"):
    st.dataframe(df)

def predict_command(command):
    command_vector = vectorizer.transform([command])
    prediction = model.predict(command_vector)[0]
    confidence = max(model.predict_proba(command_vector)[0])
    return prediction, confidence

st.subheader("Enter Aerospace Command")

user_command = st.text_input(
    "Command",
    placeholder="activate autopilot"
)

if st.button("Predict Action"):

    if user_command.strip():

        action, confidence = predict_command(
            user_command.lower()
        )

        st.success(f"Predicted Action: {action}")
        st.progress(float(confidence))
        st.info(
            f"Confidence Score: {confidence*100:.2f}%"
        )

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append(
            [user_command, action]
        )

if "history" in st.session_state and len(st.session_state.history) > 0:

    st.subheader("Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history,
        columns=["Command", "Predicted Action"]
    )

    st.dataframe(history_df)

st.subheader("Aerospace Commands Visualization")

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(df["command"], range(len(df)))

plt.xticks(rotation=90)
plt.tight_layout()

st.pyplot(fig)

st.subheader("Available Commands")
st.table(df)

st.markdown("---")
st.markdown(
    "Developed using Streamlit, Scikit-Learn and Machine Learning 🚀"
)