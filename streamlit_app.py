import streamlit as st
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from typing import Dict
import json

# MongoDB Connection
MONGO_URI = st.secrets["MONGO"]["MONGO_URI"]
client = MongoClient(MONGO_URI)
db = client["test"]
orders_collection = db["orders"]

# Page Configuration
st.set_page_config(
    page_title="Orders Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Metric",
    ["Order Count", "Referral Analysis", "Discount Code Analysis", "Order Type Analysis", "Financial & Timing Analysis", "User Analysis", "Configuration Update"]
)

# Date Range Input in Sidebar
st.sidebar.title("Date Range")
start_date = st.sidebar.date_input("Start Date", value=datetime(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now().date())

# Convert to datetime format
start_datetime = datetime.combine(start_date, datetime.min.time())
end_datetime = datetime.combine(end_date, datetime.max.time())

# Helper function to get data
def get_orders_in_date_range():
    return orders_collection.find({
        "createdAt": {"$gte": start_datetime, "$lte": end_datetime}
    })

# Helper function to process time series data
def process_time_series_data(time_df, date_column, value_column, category_column):
    # Create a complete date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_range_str = [d.strftime('%Y-%m-%d') for d in date_range]
    
    # Get unique categories
    categories = time_df[category_column].unique()
    
    # Create a complete DataFrame with all combinations
    complete_data = []
    for date in date_range_str:
        for category in categories:
            complete_data.append({
                date_column: date,
                category_column: category,
                value_column: 0
            })
    
    complete_df = pd.DataFrame(complete_data)
    
    # Merge with actual data
    merged_df = pd.merge(
        complete_df,
        time_df,
        on=[date_column, category_column],
        how='left',
        suffixes=('', '_actual')
    )
    
    # Fill NaN values with 0
    merged_df[value_column] = merged_df[f'{value_column}_actual'].fillna(0)
    merged_df = merged_df[[date_column, category_column, value_column]]
    
    return merged_df

# Function to fetch exchange rate for a specific date
def get_exchange_rate(date: datetime) -> float:
    # Format date as YYYY-MM-DD
    date_str = date.strftime('%Y-%m-%d')
    
    # Use Frankfurter API (free, no API key required)
    url = f"https://api.frankfurter.app/{date_str}?from=EUR&to=USD"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Debug the response
        if 'rates' not in data:
            st.error(f"Unexpected API response format: {data}")
            return 1.0
            
        if 'USD' not in data['rates']:
            st.error(f"USD rate not found in response: {data}")
            return 1.0
            
        return float(data['rates']['USD'])
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching exchange rate: {str(e)}")
        return 1.0
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON response from API: {str(e)}")
        return 1.0
    except Exception as e:
        st.error(f"Error fetching exchange rate: {str(e)}")
        return 1.0  # Fallback to 1:1 if API fails

# Cache for exchange rates to avoid repeated API calls
@st.cache_data(ttl=None)  # Cache indefinitely for historical rates
def get_cached_exchange_rate(date: datetime) -> float:
    # The cache key will be the date string
    # If the rate is already cached, it will return the cached value
    # If not cached, it will call get_exchange_rate and cache the result
    return get_exchange_rate(date)

# Order Count Page
if page == "Order Count":
    st.title("Order Count Analysis")
    
    # Get total order count
    order_count = orders_collection.count_documents({
        "createdAt": {"$gte": start_datetime, "$lte": end_datetime}
    })
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Orders", order_count)
    
    # Get daily order counts
    pipeline = [
        {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    daily_data = list(orders_collection.aggregate(pipeline))
    
    if daily_data:
        df = pd.DataFrame(daily_data)
        df.rename(columns={"_id": "Date", "count": "Order Count"}, inplace=True)
        
        # Line Chart
        st.subheader("Daily Order Trends")
        fig_line = px.line(df, x="Date", y="Order Count", title="Order Count Over Time")
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Bar Chart
        st.subheader("Daily Order Distribution")
        fig_bar = px.bar(df, x="Date", y="Order Count", title="Daily Order Count")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Data Table
        st.subheader("Detailed Data")
        st.dataframe(df, use_container_width=True)

# Referral Analysis Page
elif page == "Referral Analysis":
    st.title("Referral Code Analysis")
    
    pipeline = [
        {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
        {"$group": {"_id": "$referralCode", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    referral_data = list(orders_collection.aggregate(pipeline))
    
    if referral_data:
        df = pd.DataFrame(referral_data)
        df.rename(columns={"_id": "Referral Code", "count": "Order Count"}, inplace=True)
        
        # Replace empty strings with "No Referral Code"
        df["Referral Code"] = df["Referral Code"].replace("", "No Referral Code")
        
        # Get most used code excluding "No Referral Code"
        most_used_code = df[df["Referral Code"] != "No Referral Code"].iloc[0]["Referral Code"] if len(df[df["Referral Code"] != "No Referral Code"]) > 0 else "N/A"
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Referral Codes", len(df[df["Referral Code"] != "No Referral Code"]))
        with col2:
            st.metric("Most Used Code", most_used_code)
        
        # Bar Chart
        st.subheader("Orders by Referral Code")
        fig_bar = px.bar(df, x="Referral Code", y="Order Count", title="Total Orders by Referral Code")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Pie Chart
        st.subheader("Referral Code Distribution")
        fig_pie = px.pie(df, values="Order Count", names="Referral Code", title="Referral Code Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Time Series Analysis
        st.subheader("Referral Code Usage Over Time")
        
        # Get daily data for each referral code
        pipeline_time = [
            {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
            {"$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
                    "referralCode": "$referralCode"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.date": 1}}
        ]
        time_data = list(orders_collection.aggregate(pipeline_time))
        
        if time_data:
            time_df = pd.DataFrame(time_data)
            time_df["Date"] = time_df["_id"].apply(lambda x: x["date"])
            time_df["Referral Code"] = time_df["_id"].apply(lambda x: x["referralCode"])
            time_df["Referral Code"] = time_df["Referral Code"].replace("", "No Referral Code")
            time_df = time_df[["Date", "Referral Code", "count"]]
            time_df.rename(columns={"count": "Order Count"}, inplace=True)
            
            # Process time series data
            processed_df = process_time_series_data(time_df, "Date", "Order Count", "Referral Code")
            
            # Create multiselect for referral codes
            selected_codes = st.multiselect(
                "Select Referral Codes to Display",
                options=sorted(processed_df["Referral Code"].unique()),
                default=[most_used_code] if most_used_code != "N/A" else []
            )
            
            if selected_codes:
                filtered_df = processed_df[processed_df["Referral Code"].isin(selected_codes)]
                fig_time = px.line(
                    filtered_df,
                    x="Date",
                    y="Order Count",
                    color="Referral Code",
                    title="Daily Orders by Referral Code",
                    markers=True
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.write("Please select at least one referral code to display")
        
        # Data Table
        st.subheader("Detailed Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No orders found for the selected date range.")

# Discount Code Analysis Page
elif page == "Discount Code Analysis":
    st.title("Discount Code Analysis")
    
    pipeline = [
        {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
        {"$group": {"_id": "$discountCode", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    discount_data = list(orders_collection.aggregate(pipeline))
    
    if discount_data:
        df = pd.DataFrame(discount_data)
        df.rename(columns={"_id": "Discount Code", "count": "Order Count"}, inplace=True)
        
        # Replace empty strings with "No Discount Code"
        df["Discount Code"] = df["Discount Code"].replace("", "No Discount Code")
        
        # Get most used code excluding "No Discount Code"
        most_used_code = df[df["Discount Code"] != "No Discount Code"].iloc[0]["Discount Code"] if len(df[df["Discount Code"] != "No Discount Code"]) > 0 else "N/A"
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Discount Codes", len(df[df["Discount Code"] != "No Discount Code"]))
        with col2:
            st.metric("Most Used Code", most_used_code)
        
        # Bar Chart
        st.subheader("Orders by Discount Code")
        fig_bar = px.bar(df, x="Discount Code", y="Order Count", title="Total Orders by Discount Code")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Pie Chart
        st.subheader("Discount Code Distribution")
        fig_pie = px.pie(df, values="Order Count", names="Discount Code", title="Discount Code Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Time Series Analysis
        st.subheader("Discount Code Usage Over Time")
        
        # Get daily data for each discount code
        pipeline_time = [
            {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
            {"$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
                    "discountCode": "$discountCode"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.date": 1}}
        ]
        time_data = list(orders_collection.aggregate(pipeline_time))
        
        if time_data:
            time_df = pd.DataFrame(time_data)
            time_df["Date"] = time_df["_id"].apply(lambda x: x["date"])
            time_df["Discount Code"] = time_df["_id"].apply(lambda x: x["discountCode"])
            time_df["Discount Code"] = time_df["Discount Code"].replace("", "No Discount Code")
            time_df = time_df[["Date", "Discount Code", "count"]]
            time_df.rename(columns={"count": "Order Count"}, inplace=True)
            
            # Process time series data
            processed_df = process_time_series_data(time_df, "Date", "Order Count", "Discount Code")
            
            # Create multiselect for discount codes
            selected_codes = st.multiselect(
                "Select Discount Codes to Display",
                options=sorted(processed_df["Discount Code"].unique()),
                default=[most_used_code] if most_used_code != "N/A" else []
            )
            
            if selected_codes:
                filtered_df = processed_df[processed_df["Discount Code"].isin(selected_codes)]
                fig_time = px.line(
                    filtered_df,
                    x="Date",
                    y="Order Count",
                    color="Discount Code",
                    title="Daily Orders by Discount Code",
                    markers=True
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.write("Please select at least one discount code to display")
        
        # Data Table
        st.subheader("Detailed Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No orders found for the selected date range.")

# Order Type Analysis Page
elif page == "Order Type Analysis":
    st.title("Order Type Analysis")
    
    pipeline = [
        {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
        {"$group": {"_id": "$type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    type_data = list(orders_collection.aggregate(pipeline))
    
    if type_data:
        df = pd.DataFrame(type_data)
        df.rename(columns={"_id": "Order Type", "count": "Order Count"}, inplace=True)
        
        # Get most common type
        most_common_type = df.iloc[0]["Order Type"]
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Order Types", len(df))
        with col2:
            st.metric("Most Common Type", most_common_type)
        
        # Bar Chart
        st.subheader("Orders by Type")
        fig_bar = px.bar(df, x="Order Type", y="Order Count", title="Total Orders by Type")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Pie Chart
        st.subheader("Order Type Distribution")
        fig_pie = px.pie(df, values="Order Count", names="Order Type", title="Order Type Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Time Series Analysis
        st.subheader("Order Type Usage Over Time")
        
        # Get daily data for each order type
        pipeline_time = [
            {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
            {"$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
                    "type": "$type"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.date": 1}}
        ]
        time_data = list(orders_collection.aggregate(pipeline_time))
        
        if time_data:
            time_df = pd.DataFrame(time_data)
            time_df["Date"] = time_df["_id"].apply(lambda x: x["date"])
            time_df["Order Type"] = time_df["_id"].apply(lambda x: x["type"])
            time_df = time_df[["Date", "Order Type", "count"]]
            time_df.rename(columns={"count": "Order Count"}, inplace=True)
            
            # Process time series data
            processed_df = process_time_series_data(time_df, "Date", "Order Count", "Order Type")
            
            # Create multiselect for order types
            selected_types = st.multiselect(
                "Select Order Types to Display",
                options=sorted(processed_df["Order Type"].unique()),
                default=[most_common_type]
            )
            
            if selected_types:
                filtered_df = processed_df[processed_df["Order Type"].isin(selected_types)]
                fig_time = px.line(
                    filtered_df,
                    x="Date",
                    y="Order Count",
                    color="Order Type",
                    title="Daily Orders by Type",
                    markers=True
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.write("Please select at least one order type to display")
        
        # Data Table
        st.subheader("Detailed Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No orders found for the selected date range.")

# Financial & Timing Analysis Page
elif page == "Financial & Timing Analysis":
    st.title("Financial & Timing Analysis")
    
    # Get the data for the selected date range
    pipeline = [
        {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
        {"$project": {
            "priceInUSD": 1,
            "createdAt": 1,
            "updatedAt": 1,
            "orderId": 1,
            "cheapestSupplierCost": 1,
            "quantity": 1,
            "OrderDiscountValueInUSD": 1,
            "walletUsed": 1,
            "privateSupplier": 1
        }}
    ]
    data = list(orders_collection.aggregate(pipeline))
    
    if data:
        df = pd.DataFrame(data)
        
        # Ensure privateSupplier is boolean
        df['privateSupplier'] = df['privateSupplier'].fillna(False).astype(bool)
        
        # Add toggle for private supplier estimation
        include_private = st.checkbox("Include Private Supplier Estimate in Profit", value=False)
        
        # Convert supplier cost from EUR to USD
        df['exchange_rate'] = df['createdAt'].apply(get_cached_exchange_rate)
        df['supplier_cost_usd'] = df['cheapestSupplierCost'] * df['exchange_rate']
        
        if include_private:
            # Calculate monthly average supplier cost for non-private orders
            df['month'] = pd.to_datetime(df['createdAt']).dt.to_period('M')
            monthly_avg_cost = df[~df['privateSupplier']].groupby('month')['supplier_cost_usd'].mean()
            
            # Fill private supplier costs with monthly average
            for month in monthly_avg_cost.index:
                mask = (df['month'] == month) & (df['privateSupplier'])
                if monthly_avg_cost[month] > 0:  # Only fill if we have data for that month
                    df.loc[mask, 'supplier_cost_usd'] = monthly_avg_cost[month]
            
            # Calculate profit for all orders
            df["cost"] = (df["supplier_cost_usd"] * (df["quantity"]/100)) + df["OrderDiscountValueInUSD"] + df["walletUsed"]
            df["profit"] = df["priceInUSD"] - df["cost"]
            
            # Add note about private supplier estimation
            st.info("Private supplier orders are included in profit calculations using monthly average supplier costs.")
        else:
            # Calculate profit only for non-private orders
            df["cost"] = (df["supplier_cost_usd"] * (df["quantity"]/100)) + df["OrderDiscountValueInUSD"] + df["walletUsed"]
            df["profit"] = df["priceInUSD"] - df["cost"]
            # Set profit to 0 for private orders
            df.loc[df['privateSupplier'], 'profit'] = 0
            
            # Add note about excluded orders
            private_count = df['privateSupplier'].sum()
            if private_count > 0:
                st.info(f"{private_count} private supplier orders are excluded from profit calculations.")
        
        # Calculate metrics
        total_revenue = df["priceInUSD"].sum()
        if include_private:
            total_profit = df["profit"].sum()
            avg_profit = df["profit"].mean()
        else:
            # Filter out private orders before calculating profit metrics
            non_private_mask = ~df['privateSupplier']
            total_profit = df.loc[non_private_mask, "profit"].sum()
            avg_profit = df.loc[non_private_mask, "profit"].mean()
        
        avg_order_value = df["priceInUSD"].mean()
        profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
        
        # Calculate completion time in minutes
        df["completion_time"] = (pd.to_datetime(df["updatedAt"]) - pd.to_datetime(df["createdAt"])).dt.total_seconds() / 60
        avg_completion_time = df["completion_time"].mean()
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Revenue (USD)", f"${total_revenue:,.2f}")
        with col2:
            st.metric("Total Profit (USD)", f"${total_profit:,.2f}")
        with col3:
            st.metric("Average Order Value (USD)", f"${avg_order_value:,.2f}")
        with col4:
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
        with col5:
            st.metric("Avg Completion Time", f"{avg_completion_time:.1f} min")
        
        # Revenue and Profit Over Time
        st.subheader("Revenue and Profit Over Time")
        if include_private:
            daily_metrics = df.groupby(pd.to_datetime(df["createdAt"]).dt.date).agg({
                "priceInUSD": "sum",
                "profit": "sum"
            }).reset_index()
        else:
            non_private_mask = ~df['privateSupplier']
            daily_metrics = df[non_private_mask].groupby(pd.to_datetime(df["createdAt"]).dt.date).agg({
                "priceInUSD": "sum",
                "profit": "sum"
            }).reset_index()
        
        daily_metrics.columns = ["Date", "Revenue", "Profit"]
        
        # Process time series data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_range_str = [d.strftime('%Y-%m-%d') for d in date_range]
        complete_dates = pd.DataFrame({"Date": date_range_str})
        complete_dates["Date"] = pd.to_datetime(complete_dates["Date"]).dt.date
        daily_metrics["Date"] = pd.to_datetime(daily_metrics["Date"]).dt.date
        
        merged_metrics = pd.merge(complete_dates, daily_metrics, on="Date", how="left")
        merged_metrics[["Revenue", "Profit"]] = merged_metrics[["Revenue", "Profit"]].fillna(0)
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged_metrics["Date"],
            y=merged_metrics["Revenue"],
            name="Revenue",
            line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=merged_metrics["Date"],
            y=merged_metrics["Profit"],
            name="Profit",
            line=dict(color="green")
        ))
        fig.update_layout(
            title="Daily Revenue and Profit",
            xaxis_title="Date",
            yaxis_title="Amount (USD)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Profit Distribution
        st.subheader("Profit Distribution")
        if include_private:
            profit_df = df
        else:
            non_private_mask = ~df['privateSupplier']
            profit_df = df[non_private_mask]
        
        fig_profit = px.histogram(
            profit_df,
            x="profit",
            nbins=30,
            title="Distribution of Order Profits",
            labels={"profit": "Profit (USD)"}
        )
        st.plotly_chart(fig_profit, use_container_width=True)
        
        # Order Completion Time Distribution
        st.subheader("Order Completion Time Distribution")
        fig_completion = px.histogram(
            df,
            x="completion_time",
            nbins=30,
            title="Distribution of Order Completion Times",
            labels={"completion_time": "Completion Time (minutes)"}
        )
        st.plotly_chart(fig_completion, use_container_width=True)
        
        # AOV and Average Profit Over Time
        st.subheader("Average Order Value and Profit Over Time")
        if include_private:
            daily_avgs = df.groupby(pd.to_datetime(df["createdAt"]).dt.date).agg({
                "priceInUSD": "mean",
                "profit": "mean"
            }).reset_index()
        else:
            non_private_mask = ~df['privateSupplier']
            daily_avgs = df[non_private_mask].groupby(pd.to_datetime(df["createdAt"]).dt.date).agg({
                "priceInUSD": "mean",
                "profit": "mean"
            }).reset_index()
        
        daily_avgs.columns = ["Date", "AOV", "Average Profit"]
        
        # Process time series data for averages
        daily_avgs["Date"] = pd.to_datetime(daily_avgs["Date"]).dt.date
        merged_avgs = pd.merge(complete_dates, daily_avgs, on="Date", how="left")
        merged_avgs[["AOV", "Average Profit"]] = merged_avgs[["AOV", "Average Profit"]].fillna(0)
        
        fig_avgs = go.Figure()
        fig_avgs.add_trace(go.Scatter(
            x=merged_avgs["Date"],
            y=merged_avgs["AOV"],
            name="Average Order Value",
            line=dict(color="blue")
        ))
        fig_avgs.add_trace(go.Scatter(
            x=merged_avgs["Date"],
            y=merged_avgs["Average Profit"],
            name="Average Profit",
            line=dict(color="green")
        ))
        fig_avgs.update_layout(
            title="Daily Average Order Value and Profit",
            xaxis_title="Date",
            yaxis_title="Amount (USD)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_avgs, use_container_width=True)
        
        # Detailed Data Table
        st.subheader("Detailed Data")
        display_df = df[["orderId", "priceInUSD", "cost", "profit", "completion_time", "createdAt", "updatedAt", "privateSupplier"]]
        display_df.columns = ["Order ID", "Revenue (USD)", "Cost (USD)", "Profit (USD)", "Completion Time (min)", "Created At", "Updated At", "Private Supplier"]
        st.dataframe(display_df, use_container_width=True)
    else:
        st.write("No orders found for the selected date range.")

# User Analysis Page
elif page == "User Analysis":
    st.title("User Analysis")
    
    # Get all users with wallet information
    users_collection = db["users"]
    all_users = list(users_collection.find({}, {"email": 1, "wallet": 1}))
    user_emails = [user["email"] for user in all_users]
    
    # Get orders for the selected date range
    pipeline = [
        {"$match": {"createdAt": {"$gte": start_datetime, "$lte": end_datetime}}},
        {"$group": {
            "_id": "$contactEmail",
            "total_orders": {"$sum": 1},
            "total_quantity": {"$sum": "$quantity"},
            "total_spent": {"$sum": "$priceInUSD"}
        }},
        {"$sort": {"total_spent": -1}}
    ]
    user_orders = list(orders_collection.aggregate(pipeline))
    
    if user_orders:
        # Create DataFrame for users with orders
        orders_df = pd.DataFrame(user_orders)
        orders_df.rename(columns={"_id": "email"}, inplace=True)
        
        # Create DataFrame for all users with wallet information
        users_df = pd.DataFrame(all_users)
        users_df.rename(columns={"email": "email", "wallet": "wallet_amount"}, inplace=True)
        
        # Merge with orders data
        user_analysis_df = pd.merge(users_df, orders_df, on="email", how="left")
        
        # Fill NaN values for users without orders
        user_analysis_df = user_analysis_df.fillna({
            "total_orders": 0,
            "total_quantity": 0,
            "total_spent": 0
        })
        
        # Calculate metrics
        total_users = len(user_analysis_df)
        active_users = len(user_analysis_df[user_analysis_df["total_orders"] > 0])
        inactive_users = total_users - active_users
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", total_users)
        with col2:
            st.metric("Active Users", active_users)
        with col3:
            st.metric("Inactive Users", inactive_users)
        
        # User Status Toggle
        user_status = st.radio(
            "Show Users",
            ["All Users", "Active Users Only", "Inactive Users Only"],
            horizontal=True
        )
        
        # Filter users based on selection
        if user_status == "Active Users Only":
            filtered_df = user_analysis_df[user_analysis_df["total_orders"] > 0]
        elif user_status == "Inactive Users Only":
            filtered_df = user_analysis_df[user_analysis_df["total_orders"] == 0]
        else:
            filtered_df = user_analysis_df
        
        # Detailed User Data
        st.subheader("Detailed User Data")
        display_df = filtered_df[[
            "email", "wallet_amount", "total_orders", "total_quantity", "total_spent"
        ]]
        display_df.columns = [
            "Email", "Wallet Amount (USD)", "Total Orders", "Total Quantity", "Total Spent (USD)"
        ]
        # Sort by Total Spent
        display_df = display_df.sort_values("Total Spent (USD)", ascending=False)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.write("No order data found for the selected date range.")

# Configuration Update Page
elif page == "Configuration Update":
    st.title("Configuration Update")
    
    # Get current sell prices
    sellprices_collection = db["sellprices"]
    current_sellprices = sellprices_collection.find_one({})
    
    # Get current pricing configurations
    pricing_config_collection = db["pricing_configurations"]
    current_pricing_config = pricing_config_collection.find_one({})
    
    # Create two columns for the two sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sell Prices (USD)")
        if current_sellprices:
            # Display current values
            st.write("Current Values:")
            st.write(f"PS: ${current_sellprices.get('PS', 0):.2f}")
            st.write(f"PC: ${current_sellprices.get('PC', 0):.2f}")
            st.write(f"XB: ${current_sellprices.get('XB', 0):.2f}")
            
            # Input fields for new values
            st.write("Update Values:")
            new_ps = st.number_input("PS Price (USD)", value=float(current_sellprices.get('PS', 0.9)), min_value=0.0, step=float(0.1), format="%.2f")
            new_pc = st.number_input("PC Price (USD)", value=float(current_sellprices.get('PC', 1.2)), min_value=0.0, step=float(0.1), format="%.2f")
            new_xb = st.number_input("XB Price (USD)", value=float(current_sellprices.get('XB', 0.9)), min_value=0.0, step=float(0.1), format="%.2f")
            
            if st.button("Update Sell Prices"):
                try:
                    sellprices_collection.update_one(
                        {},
                        {"$set": {
                            "PS": new_ps,
                            "PC": new_pc,
                            "XB": new_xb
                        }},
                        upsert=True
                    )
                    st.toast("✅ Sell prices updated successfully!", icon="✅")
                except Exception as e:
                    st.toast(f"❌ Error updating sell prices: {str(e)}", icon="❌")
        else:
            st.write("No current sell prices found. Setting default values.")
            new_ps = st.number_input("PS Price (USD)", value=float(0.9), min_value=0.0, step=float(0.1), format="%.2f")
            new_pc = st.number_input("PC Price (USD)", value=float(1.2), min_value=0.0, step=float(0.1), format="%.2f")
            new_xb = st.number_input("XB Price (USD)", value=float(0.9), min_value=0.0, step=float(0.1), format="%.2f")
            
            if st.button("Set Initial Sell Prices"):
                try:
                    sellprices_collection.insert_one({
                        "PS": new_ps,
                        "PC": new_pc,
                        "XB": new_xb
                    })
                    st.toast("✅ Initial sell prices set successfully!", icon="✅")
                except Exception as e:
                    st.toast(f"❌ Error setting sell prices: {str(e)}", icon="❌")
    
    with col2:
        st.subheader("Pricing Configurations")
        if current_pricing_config:
            # Display current values
            st.write("Current Values:")
            st.write(f"Min PC Price Threshold: {current_pricing_config.get('min_pc_price_threshold', 2.0):.2f}")
            st.write(f"Min Console Price Threshold: {current_pricing_config.get('min_console_price_threshold', 2.0):.2f}")
            st.write(f"Min Stock PC: {current_pricing_config.get('min_stock_pc', 2000)}")
            st.write(f"Min Avg PC: {current_pricing_config.get('min_avg_pc', 100)}")
            st.write(f"Min Stock Console: {current_pricing_config.get('min_stock_console', 2000)}")
            st.write(f"Min Avg Console: {current_pricing_config.get('min_avg_console', 100)}")
            st.write(f"Desired Margin: {current_pricing_config.get('desired_margin', 0.3)}")
            st.write(f"Affiliate Percentage: {current_pricing_config.get('affiliate_pct', 0.0065)}")
            st.write(f"Bonus Percentage: {current_pricing_config.get('bonus_pct', 0.1)}")
            st.write(f"Discount Percentage: {current_pricing_config.get('discount_pct', 0)}")
            st.write(f"FCS Fixed: {current_pricing_config.get('fcs_fixed', 0.3)}")
            st.write(f"FCS Percentage: {current_pricing_config.get('fcs_pct', 0.04)}")
            st.write(f"Stripe Percentage: {current_pricing_config.get('stripe_pct', 0.05)}")
            st.write(f"Wallet Percentage: {current_pricing_config.get('wallet_pct', 0.1)}")
            
            # Input fields for new values
            st.write("Update Values:")
            new_min_pc_price = st.number_input("Min PC Price Threshold", value=float(current_pricing_config.get('min_pc_price_threshold', 2.0)), min_value=0.0, step=float(0.1), format="%.2f")
            new_min_console_price = st.number_input("Min Console Price Threshold", value=float(current_pricing_config.get('min_console_price_threshold', 2.0)), min_value=0.0, step=float(0.1), format="%.2f")
            new_min_stock_pc = st.number_input("Min Stock PC", value=int(current_pricing_config.get('min_stock_pc', 2000)), min_value=0, step=int(100))
            new_min_avg_pc = st.number_input("Min Avg PC", value=int(current_pricing_config.get('min_avg_pc', 100)), min_value=0, step=int(10))
            new_min_stock_console = st.number_input("Min Stock Console", value=int(current_pricing_config.get('min_stock_console', 2000)), min_value=0, step=int(100))
            new_min_avg_console = st.number_input("Min Avg Console", value=int(current_pricing_config.get('min_avg_console', 100)), min_value=0, step=int(10))
            new_desired_margin = st.number_input("Desired Margin", value=float(current_pricing_config.get('desired_margin', 0.3)), min_value=0.0, step=float(0.01), format="%.2f")
            new_affiliate_pct = st.number_input("Affiliate Percentage", value=float(current_pricing_config.get('affiliate_pct', 0.0065)), min_value=0.0, step=float(0.0001), format="%.4f")
            new_bonus_pct = st.number_input("Bonus Percentage", value=float(current_pricing_config.get('bonus_pct', 0.1)), min_value=0.0, step=float(0.01), format="%.2f")
            new_discount_pct = st.number_input("Discount Percentage", value=float(current_pricing_config.get('discount_pct', 0)), min_value=0.0, step=float(0.01), format="%.2f")
            new_fcs_fixed = st.number_input("FCS Fixed", value=float(current_pricing_config.get('fcs_fixed', 0.3)), min_value=0.0, step=float(0.01), format="%.2f")
            new_fcs_pct = st.number_input("FCS Percentage", value=float(current_pricing_config.get('fcs_pct', 0.04)), min_value=0.0, step=float(0.01), format="%.2f")
            new_stripe_pct = st.number_input("Stripe Percentage", value=float(current_pricing_config.get('stripe_pct', 0.05)), min_value=0.0, step=float(0.01), format="%.2f")
            new_wallet_pct = st.number_input("Wallet Percentage", value=float(current_pricing_config.get('wallet_pct', 0.1)), min_value=0.0, step=float(0.01), format="%.2f")
            
            if st.button("Update Pricing Configurations"):
                try:
                    pricing_config_collection.update_one(
                        {},
                        {"$set": {
                            "min_pc_price_threshold": new_min_pc_price,
                            "min_console_price_threshold": new_min_console_price,
                            "min_stock_pc": new_min_stock_pc,
                            "min_avg_pc": new_min_avg_pc,
                            "min_stock_console": new_min_stock_console,
                            "min_avg_console": new_min_avg_console,
                            "desired_margin": new_desired_margin,
                            "affiliate_pct": new_affiliate_pct,
                            "bonus_pct": new_bonus_pct,
                            "discount_pct": new_discount_pct,
                            "fcs_fixed": new_fcs_fixed,
                            "fcs_pct": new_fcs_pct,
                            "stripe_pct": new_stripe_pct,
                            "wallet_pct": new_wallet_pct
                        }},
                        upsert=True
                    )
                    st.toast("✅ Pricing configurations updated successfully!", icon="✅")
                except Exception as e:
                    st.toast(f"❌ Error updating pricing configurations: {str(e)}", icon="❌")
        else:
            st.write("No current pricing configurations found. Setting default values.")
            # Input fields for initial values
            new_min_pc_price = st.number_input("Min PC Price Threshold", value=float(2.0), min_value=0.0, step=float(0.1), format="%.2f")
            new_min_console_price = st.number_input("Min Console Price Threshold", value=float(2.0), min_value=0.0, step=float(0.1), format="%.2f")
            new_min_stock_pc = st.number_input("Min Stock PC", value=int(2000), min_value=0, step=int(100))
            new_min_avg_pc = st.number_input("Min Avg PC", value=int(100), min_value=0, step=int(10))
            new_min_stock_console = st.number_input("Min Stock Console", value=int(2000), min_value=0, step=int(100))
            new_min_avg_console = st.number_input("Min Avg Console", value=int(100), min_value=0, step=int(10))
            new_desired_margin = st.number_input("Desired Margin", value=float(0.3), min_value=0.0, step=float(0.01), format="%.2f")
            new_affiliate_pct = st.number_input("Affiliate Percentage", value=float(0.0065), min_value=0.0, step=float(0.0001), format="%.4f")
            new_bonus_pct = st.number_input("Bonus Percentage", value=float(0.1), min_value=0.0, step=float(0.01), format="%.2f")
            new_discount_pct = st.number_input("Discount Percentage", value=float(0), min_value=0.0, step=float(0.01), format="%.2f")
            new_fcs_fixed = st.number_input("FCS Fixed", value=float(0.3), min_value=0.0, step=float(0.01), format="%.2f")
            new_fcs_pct = st.number_input("FCS Percentage", value=float(0.04), min_value=0.0, step=float(0.01), format="%.2f")
            new_stripe_pct = st.number_input("Stripe Percentage", value=float(0.05), min_value=0.0, step=float(0.01), format="%.2f")
            new_wallet_pct = st.number_input("Wallet Percentage", value=float(0.1), min_value=0.0, step=float(0.01), format="%.2f")
            
            if st.button("Set Initial Pricing Configurations"):
                try:
                    pricing_config_collection.insert_one({
                        "min_pc_price_threshold": new_min_pc_price,
                        "min_console_price_threshold": new_min_console_price,
                        "min_stock_pc": new_min_stock_pc,
                        "min_avg_pc": new_min_avg_pc,
                        "min_stock_console": new_min_stock_console,
                        "min_avg_console": new_min_avg_console,
                        "desired_margin": new_desired_margin,
                        "affiliate_pct": new_affiliate_pct,
                        "bonus_pct": new_bonus_pct,
                        "discount_pct": new_discount_pct,
                        "fcs_fixed": new_fcs_fixed,
                        "fcs_pct": new_fcs_pct,
                        "stripe_pct": new_stripe_pct,
                        "wallet_pct": new_wallet_pct
                    })
                    st.toast("✅ Initial pricing configurations set successfully!", icon="✅")
                except Exception as e:
                    st.toast(f"❌ Error setting pricing configurations: {str(e)}", icon="❌")