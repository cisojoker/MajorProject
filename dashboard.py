import streamlit as st
import pandas as pd
import hashlib
import plotly.express as px
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Access the Mongo URI from environment variable
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client['Users_Credentials']
collection = db['Users_Credentials']

def load_user_data():
    try:
        users = list(collection.find({}, {"_id": 0}))
        for user in users:
            if isinstance(user.get("signup_timestamp"), datetime):
                user["signup_timestamp"] = pd.to_datetime(user["signup_timestamp"], errors='coerce')
            else:
                user["signup_timestamp"] = pd.NaT

            login_history = user.get("login_history", [])
            if isinstance(login_history, list) and login_history:
                user["total_logins"] = len(login_history)
                user["last_login"] = pd.to_datetime(max(login_history), errors='coerce')
            else:
                user["total_logins"] = 0
                user["last_login"] = pd.NaT

            user["timestamp"] = user["signup_timestamp"] if pd.notna(user["signup_timestamp"]) else user["last_login"]

        df = pd.DataFrame(users)
        return df.dropna(subset=["username"])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- Streamlit page setup ---
st.set_page_config("User Management Dashboard", layout="wide")

st.markdown("""
    <style>
        .gradient-text {
            font-size: 48px;
            font-weight: bold;
            background: linear-gradient(90deg, #1995AD, #A1C6EA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>

    <h1 class='gradient-text'>👥 User Management Dashboard</h1>
    <p style='color:#555;'>Manage user accounts, detect weak security, and review account activity.</p>
""", unsafe_allow_html=True)

# --- Load & Clean Data ---
df = load_user_data()
df = df.dropna(subset=["timestamp"])
df = df.reset_index(drop=True)

# --- Sidebar Filters ---
st.sidebar.header("🔍 Search & Filters")
search_term = st.sidebar.text_input("Search username:")
st.sidebar.subheader("📅 Date Filters")

if not df.empty:
    available_years = sorted(df["timestamp"].dt.year.dropna().unique().tolist())
else:
    available_years = [datetime.now().year]

available_months = list(range(1, 13))
selected_year = st.sidebar.selectbox("Filter by Year:", ["All Years"] + available_years, index=0)
selected_month = st.sidebar.selectbox("Filter by Month:", ["All Months"] + [datetime(2000, m, 1).strftime('%B') for m in available_months], index=0)

# --- Apply Filters ---
filtered_df = df.copy()

if search_term:
    filtered_df = filtered_df[filtered_df["username"].str.contains(search_term, case=False, na=False)]

if selected_year != "All Years":
    signup_filter = filtered_df["signup_timestamp"].dt.year == selected_year
    login_filter = filtered_df["last_login"].dt.year == selected_year
    filtered_df = filtered_df[signup_filter | login_filter.fillna(False)]

if selected_month != "All Months":
    month_num = datetime.strptime(selected_month, '%B').month
    signup_filter = filtered_df["signup_timestamp"].dt.month == month_num
    login_filter = filtered_df["last_login"].dt.month == month_num
    filtered_df = filtered_df[signup_filter | login_filter.fillna(False)]

# --- Summary Metrics ---
st.subheader("📊 Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("👤 Total Users", filtered_df.shape[0])
with col2:
    st.metric("🔑 Unique Passwords", filtered_df["hashed_password"].nunique() if "hashed_password" in filtered_df else '-')
with col3:
    reused_count = filtered_df["hashed_password"].duplicated(keep=False).sum() if "hashed_password" in filtered_df else 0
    st.metric("⚠️ Passwords Reused", reused_count)
with col4:
    total_logins = filtered_df["total_logins"].sum() if "total_logins" in filtered_df else 0
    st.metric("🔄 Total Logins", int(total_logins))

# --- Recent User Activity ---
st.subheader("🕒 Recent User Activity")
if "timestamp" in filtered_df:
    recent_users = filtered_df.sort_values(by="timestamp", ascending=False).head(5)[["username", "timestamp"]]
    st.table(recent_users)

# --- Tabs View ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Users", "⚠️ Reused Passwords", "📈 Reuse Frequency", "📅 User Growth", "🔄 Login Activity"])

with tab1:
    st.subheader("📋 User List")
    if not filtered_df.empty:
        display_df = filtered_df[["username", "signup_timestamp", "last_login", "total_logins"]].copy()
        display_df.columns = ["Username", "Signup Time", "Last Login", "Total Logins"]
        st.dataframe(display_df, use_container_width=True)

with tab2:
    st.subheader("⚠️ Users with Reused Passwords")
    reused_df = filtered_df[filtered_df.duplicated(subset="hashed_password", keep=False)] if "hashed_password" in filtered_df else pd.DataFrame()
    if reused_df.empty:
        st.success("✅ All users have unique passwords!")
    else:
        st.warning("Some users are sharing the same password (hash hidden).")
        grouped = reused_df.groupby("hashed_password")["username"].apply(list).reset_index()
        grouped["User Count"] = grouped["username"].apply(len)
        grouped["Users"] = grouped["username"].apply(lambda x: ", ".join(x))
        st.dataframe(grouped[["Users", "User Count"]], use_container_width=True)

with tab3:
    st.subheader("🔢 Password Reuse Frequency (Anonymized)")
    if reused_df.empty:
        st.info("No reused passwords to display.")
    else:
        reuse_counts = reused_df["hashed_password"].value_counts().reset_index()
        reuse_counts.columns = ["Anonymized Password", "User Count"]
        reuse_counts["Anonymized Password"] = reuse_counts["Anonymized Password"].apply(lambda x: "hash_" + str(hash(x))[-6:])
        fig = px.bar(
            reuse_counts,
            x="Anonymized Password",
            y="User Count",
            title="Password Reuse (Anonymized)",
            color="User Count",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("📅 User Signup Growth")
    if filtered_df["timestamp"].isnull().all():
        st.warning("Timestamps not available for user signups.")
    else:
        filtered_df["date"] = filtered_df["timestamp"].dt.date
        signup_trend = filtered_df.groupby('date').size().reset_index(name='User Count')

        title = 'User Signups Over Time'
        if selected_year != "All Years" or selected_month != "All Months":
            filter_text = []
            if selected_month != "All Months":
                filter_text.append(selected_month)
            if selected_year != "All Years":
                filter_text.append(str(selected_year))
            title += f" ({' '.join(filter_text)})"

        fig = px.line(signup_trend, x='date', y='User Count', title=title, markers=True)
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("🔄 Login Activity Analysis")

    st.write("**Most Active Users:**")
    if "total_logins" in filtered_df:
        active_users = filtered_df.nlargest(5, 'total_logins')[["username", "total_logins", "last_login"]]
        st.dataframe(active_users, use_container_width=True)

        fig = px.histogram(
            filtered_df,
            x="total_logins",
            title="Distribution of Login Frequency",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)


