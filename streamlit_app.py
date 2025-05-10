import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import plotly.express as px

# MongoDB Connection
MONGO_URI = st.secrets["MONGO"]["MONGO_URI"]
client = MongoClient(MONGO_URI)
db = client["test"]
orders_collection = db["orders"]

# Streamlit App Title
st.title("Orders Analysis by Date Range")

# Date Range Input
start_date = st.date_input("Start Date", value=datetime(2025, 1, 1))
end_date = st.date_input("End Date", value=datetime(2025, 1, 31))

# Convert to datetime format
start_datetime = datetime.combine(start_date, datetime.min.time())
end_datetime = datetime.combine(end_date, datetime.max.time())

# Button for order count
if st.button("Get Order Count"):
    order_count = orders_collection.count_documents({
        "createdAt": {"$gte": start_datetime, "$lte": end_datetime}
    })
    st.write(f"Total Orders from {start_date} to {end_date}: **{order_count}**")

# Button for referral analysis
if st.button("Get Referral Analysis"):
    pipeline = [
        {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
        {"$group": {"_id": "$referralCode", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    referral_data = list(orders_collection.aggregate(pipeline))
    
    if referral_data:
        df = pd.DataFrame(referral_data)
        df.rename(columns={"_id": "Referral Code", "count": "Order Count"}, inplace=True)
        
        # Display Bar Chart
        fig = px.bar(df, x="Referral Code", y="Order Count", title="Total Orders by Referral Code", text="Order Count")
        st.plotly_chart(fig)
    else:
        st.write("No orders found for the selected date range.")

if st.button("Get Order Count by Discount Code"):
    # Aggregate discount code counts
    pipeline = [
        {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
        {"$group": {"_id": "$discountCode", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    discount_data = list(orders_collection.aggregate(pipeline))
    
    if discount_data:
        df = pd.DataFrame(discount_data)
        df.rename(columns={"_id": "Discount Code", "count": "Order Count"}, inplace=True)
        
        # Display Bar Chart
        fig = px.bar(df, x="Discount Code", y="Order Count", title="Total Orders by Discount Code", text="Order Count")
        st.plotly_chart(fig)
    else:
        st.write("No orders found for the selected date range.")

if st.button("Get Order Count by Type"):
    # Aggregate type counts
    pipeline = [
        {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
        {"$group": {"_id": "$type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    type_data = list(orders_collection.aggregate(pipeline))
    
    if type_data:
        df = pd.DataFrame(type_data)
        df.rename(columns={"_id": "Order Type", "count": "Order Count"}, inplace=True)
        
        # Display Bar Chart
        fig = px.bar(df, x="Order Type", y="Order Count", title="Total Orders by Type", text="Order Count")
        st.plotly_chart(fig)
    else:
        st.write("No orders found for the selected date range.")