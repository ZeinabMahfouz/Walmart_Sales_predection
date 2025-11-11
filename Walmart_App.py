
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    h1 {color: #0066cc; font-family: 'Helvetica Neue', sans-serif; font-weight: 700;}
    h2 {color: #004080; font-family: 'Helvetica Neue', sans-serif;}
    h3 {color: #005cb3;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 10px; color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px; border-radius: 15px; color: white;
        text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        font-size: 24px; font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd; padding: 15px; border-radius: 10px;
        border-left: 5px solid #2196f3; margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9; padding: 15px; border-radius: 10px;
        border-left: 5px solid #4caf50; margin: 10px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: bold; border-radius: 10px;
        border: none; padding: 10px 30px; font-size: 16px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================

@st.cache_data
def load_data():
    """Load the Walmart cleaned dataset"""
    try:
        df = pd.read_csv('walmart_cleaned.csv')
        st.sidebar.success("✓ Loaded walmart_cleaned.csv")
        
        # Convert Date if present
        if 'Date' in df.columns and df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    except FileNotFoundError:
        st.sidebar.error("❌ walmart_cleaned.csv not found!")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load trained models"""
    try:
        model = joblib.load('best_model_cv.pkl')
        scaler = joblib.load('scaler_cv.pkl')
        with open('model_info_cv.json', 'r') as f:
            model_info = json.load(f)
        
        # Get features and remove Date and Log_Weekly_Sales if present
        features = model_info.get('features', [])
        features = [f for f in features if f not in ['Date', 'Log_Weekly_Sales']]
        model_info['features'] = features
        
        st.sidebar.success("✓ Model loaded successfully")
        return model, scaler, model_info
    except Exception as e:
        st.sidebar.error(f"❌ Model error: {e}")
        return None, None, None

# Load data
df = load_data()
model, scaler, model_info = load_model_artifacts()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/ca/Walmart_logo.svg", width=200)
st.sidebar.title("🛒 Navigation")

page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📊 Data Explorer", "🔮 Make Predictions", "📈 Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info(
    f"""
    **Walmart Sales Prediction**
    
    Predict weekly sales using ML
    
    **Dataset:** 45 stores (2010-2012)
    
    **Model:** {model_info['best_model'] if model_info else 'Loading...'}
    
    **Accuracy:** {model_info['test_r2']*100:.1f}% if model_info else 'N/A'
    """
)

# ============================================================================
# PAGE 1: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        border-radius: 15px; margin-bottom: 30px;'>
            <h1 style='color: white; font-size: 48px; margin: 0;'>🛒 Walmart Sales Predictor</h1>
            <p style='color: white; font-size: 20px; margin-top: 10px;'>
                ML-Powered Retail Forecasting System
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if df is not None and model_info is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0; color: white;'>📦 Records</h3>
                    <h2 style='margin: 10px 0 0 0; color: white;'>{len(df):,}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0; color: white;'>🏪 Stores</h3>
                    <h2 style='margin: 10px 0 0 0; color: white;'>{df['Store'].nunique()}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0; color: white;'>🎯 Accuracy</h3>
                    <h2 style='margin: 10px 0 0 0; color: white;'>{model_info['test_r2']*100:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='margin: 0; color: white;'>💰 Avg Sales</h3>
                    <h2 style='margin: 10px 0 0 0; color: white;'>${df['Weekly_Sales'].mean():,.0f}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 📋 Project Overview")
            st.markdown(f"""
                <div class='info-box'>
                    <h3>🎯 Objective</h3>
                    <p>Predict weekly sales for Walmart stores using historical data and various features.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class='info-box'>
                    <h3>📊 Features Used</h3>
                    <p><strong>{len(model_info['features'])}</strong> predictive features: {', '.join(model_info['features'][:5])}...</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("## 🏆 Best Model")
            st.markdown(f"""
                <div class='success-box'>
                    <h3 style='color: #2e7d32;'>{model_info['best_model']}</h3>
                    <p><strong>Test R²:</strong> {model_info['test_r2']:.4f}</p>
                    <p><strong>Test RMSE:</strong> ${model_info['test_rmse']:,.2f}</p>
                    <p><strong>CV Folds:</strong> {model_info['cv_folds']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Quick Stats
        st.markdown("## 📊 Quick Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Sales", f"${df['Weekly_Sales'].sum()/1e9:.2f}B")
        with col2:
            st.metric("Date Range", f"{df['Year'].min()} - {df['Year'].max()}")
        with col3:
            st.metric("Holiday Weeks", f"{df['Holiday_Flag'].sum()} ({df['Holiday_Flag'].mean()*100:.1f}%)")

# ============================================================================
# PAGE 2: DATA EXPLORER
# ============================================================================

elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📋 Data View", "📈 Visualizations", "🔍 Statistics"])
        
        with tab1:
            st.markdown("### 🔍 Dataset Preview")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_stores = st.multiselect(
                    "Filter by Store",
                    options=sorted(df['Store'].unique()),
                    default=None
                )
            with col2:
                selected_years = st.multiselect(
                    "Filter by Year",
                    options=sorted(df['Year'].unique()),
                    default=None
                )
            with col3:
                holiday_filter = st.selectbox(
                    "Holiday Filter",
                    options=["All", "Holiday Only", "Non-Holiday Only"]
                )
            
            # Apply filters
            filtered_df = df.copy()
            if selected_stores:
                filtered_df = filtered_df[filtered_df['Store'].isin(selected_stores)]
            if selected_years:
                filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
            if holiday_filter == "Holiday Only":
                filtered_df = filtered_df[filtered_df['Holiday_Flag'] == 1]
            elif holiday_filter == "Non-Holiday Only":
                filtered_df = filtered_df[filtered_df['Holiday_Flag'] == 0]
            
            st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} records**")
            
            # Display columns (exclude Log_Weekly_Sales and Date if present)
            display_cols = [col for col in filtered_df.columns if col not in ['Date', 'Log_Weekly_Sales']]
            st.dataframe(filtered_df[display_cols].head(100), height=400)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="walmart_filtered_data.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.markdown("### 📈 Interactive Visualizations")
            
            # Sales over time
            fig1 = px.line(df.sort_values('WeekOfYear'), 
                          x='WeekOfYear', y='Weekly_Sales',
                          title='Weekly Sales Pattern (by Week of Year)',
                          labels={'WeekOfYear': 'Week of Year', 'Weekly_Sales': 'Sales ($)'})
            fig1.update_traces(line_color='#667eea', line_width=2)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Store comparison
            col1, col2 = st.columns(2)
            
            with col1:
                store_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False).head(10)
                fig2 = px.bar(x=store_sales.index, y=store_sales.values,
                             title='Top 10 Stores by Average Sales',
                             labels={'x': 'Store', 'y': 'Average Sales ($)'})
                fig2.update_traces(marker_color='#764ba2')
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
                fig3 = px.line(x=monthly_sales.index, y=monthly_sales.values,
                              title='Average Sales by Month',
                              labels={'x': 'Month', 'y': 'Average Sales ($)'},
                              markers=True)
                fig3.update_traces(line_color='#f093fb', marker_size=10)
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            st.markdown("### 📊 Statistical Summary")
            
            st.dataframe(df.describe(), use_container_width=True)
            
            st.markdown("### 🔢 Feature Correlations")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['Date', 'Log_Weekly_Sales']]
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix,
                           labels=dict(color="Correlation"),
                           x=corr_matrix.columns,
                           y=corr_matrix.columns,
                           color_continuous_scale='RdBu_r',
                           title='Feature Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: MAKE PREDICTIONS
# ============================================================================

elif page == "🔮 Make Predictions":
    st.title("🔮 Make Sales Predictions")
    
    if model is not None and scaler is not None and model_info is not None:
        
        tab1, tab2 = st.tabs(["📝 Single Prediction", "📁 Batch Prediction"])
        
        with tab1:
            st.markdown("""
                <div class='info-box'>
                    <h3>👇 Enter Store Details</h3>
                    <p>Fill in the features below to predict weekly sales.</p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    store = st.number_input("Store Number", min_value=1, max_value=45, value=1)
                    holiday_flag = st.selectbox("Holiday Week?", options=[0, 1], 
                                               format_func=lambda x: "No" if x == 0 else "Yes")
                    temperature = st.number_input("Temperature (°F)", min_value=20.0, max_value=120.0, value=60.0)
                    fuel_price = st.number_input("Fuel Price ($/gal)", min_value=2.0, max_value=5.0, value=3.0, step=0.01)
                
                with col2:
                    cpi = st.number_input("CPI", min_value=180.0, max_value=250.0, value=211.0, step=0.1)
                    unemployment = st.number_input("Unemployment (%)", min_value=2.0, max_value=15.0, value=8.0, step=0.1)
                    year = st.number_input("Year", min_value=2010, max_value=2030, value=2024)
                    month = st.selectbox("Month", options=list(range(1, 13)), index=10)
                
                with col3:
                    day = st.number_input("Day", min_value=1, max_value=31, value=15)
                    quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=3)
                    is_weekend = st.selectbox("Is Weekend?", options=[0, 1], 
                                             format_func=lambda x: "No" if x == 0 else "Yes")
                    week_of_year = st.number_input("Week of Year", min_value=1, max_value=53, value=46)
                
                submit_button = st.form_submit_button("🔮 Predict Sales")
            
            if submit_button:
                # Prepare input
                input_data = pd.DataFrame({
                    'Store': [store],
                    'Holiday_Flag': [holiday_flag],
                    'Temperature': [temperature],
                    'Fuel_Price': [fuel_price],
                    'CPI': [cpi],
                    'Unemployment': [unemployment],
                    'Year': [year],
                    'Month': [month],
                    'Day': [day],
                    'Quarter': [quarter],
                    'IsWeekend': [is_weekend],
                    'WeekOfYear': [week_of_year]
                })
                
                # Match model features
                required_features = model_info['features']
                input_data = input_data[required_features]
                
                # Scale and predict
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                
                # Display prediction
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='prediction-box'>
                        💰 Predicted Weekly Sales: ${prediction:,.2f}
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Confidence", f"{model_info['test_r2']*100:.1f}%")
                
                with col2:
                    avg_sales = df['Weekly_Sales'].mean()
                    diff = ((prediction - avg_sales) / avg_sales * 100)
                    st.metric("vs Average", f"{diff:+.1f}%", delta=f"${prediction - avg_sales:,.2f}")
                
                with col3:
                    store_avg = df[df['Store'] == store]['Weekly_Sales'].mean()
                    store_diff = ((prediction - store_avg) / store_avg * 100)
                    st.metric(f"vs Store {store} Avg", f"{store_diff:+.1f}%", delta=f"${prediction - store_avg:,.2f}")
        
        with tab2:
            st.markdown("## 📁 Batch Prediction")
            st.markdown("Upload a CSV file with the required columns to get bulk predictions.")
            
            st.markdown(f"""
                <div class='info-box'>
                    <strong>Required columns:</strong> {', '.join(model_info['features'])}
                </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file:
                try:
                    batch_data = pd.read_csv(uploaded_file)
                    st.markdown("### 📋 Uploaded Data")
                    st.dataframe(batch_data.head(10))
                    
                    if st.button("🔮 Generate Predictions"):
                        # Match features
                        required_features = model_info['features']
                        
                        # Check missing columns
                        missing = [col for col in required_features if col not in batch_data.columns]
                        if missing:
                            st.error(f"❌ Missing columns: {missing}")
                        else:
                            batch_input = batch_data[required_features]
                            batch_scaled = scaler.transform(batch_input)
                            predictions = model.predict(batch_scaled)
                            
                            batch_data['Predicted_Sales'] = predictions
                            
                            st.markdown("### ✅ Predictions Complete!")
                            st.dataframe(batch_data)
                            
                            # Download
                            csv = batch_data.to_csv(index=False)
                            st.download_button(
                                "📥 Download Predictions",
                                data=csv,
                                file_name="walmart_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Predictions", len(predictions))
                            with col2:
                                st.metric("Avg Predicted Sales", f"${predictions.mean():,.2f}")
                            with col3:
                                st.metric("Total Predicted", f"${predictions.sum()/1e6:.2f}M")
                
                except Exception as e:
                    st.error(f"Error: {e}")

# ============================================================================
# PAGE 4: INSIGHTS
# ============================================================================

elif page == "📈 Insights":
    st.title("📈 Key Insights & Findings")
    
    if df is not None and model_info is not None:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; color: white; margin-bottom: 30px;'>
                <h2 style='margin: 0; color: white;'>🎯 Executive Summary</h2>
                <p style='font-size: 18px; margin-top: 15px;'>
                    ML model achieves <strong>{model_info['test_r2']*100:.1f}% accuracy</strong> 
                    in predicting Walmart sales using <strong>{model_info['best_model']}</strong>.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='success-box'>
                    <h3>📊 Data Insights</h3>
                    <ul>
                        <li><strong>Q4 Peak:</strong> Highest sales in Oct-Dec</li>
                        <li><strong>Holiday Effect:</strong> Variable impact</li>
                        <li><strong>Store Variance:</strong> 2-3x difference</li>
                        <li><strong>Weather Impact:</strong> Moderate correlation</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='info-box'>
                    <h3>🤖 Model Performance</h3>
                    <ul>
                        <li><strong>Algorithm:</strong> {model_info['best_model']}</li>
                        <li><strong>Test R²:</strong> {model_info['test_r2']:.4f}</li>
                        <li><strong>Test RMSE:</strong> ${model_info['test_rmse']:,.2f}</li>
                        <li><strong>Features:</strong> {len(model_info['features'])}</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("## 📊 Visual Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quarterly = df.groupby('Quarter')['Weekly_Sales'].mean()
            fig1 = px.bar(x=['Q1', 'Q2', 'Q3', 'Q4'], y=quarterly.values,
                         title='Average Sales by Quarter',
                         labels={'x': 'Quarter', 'y': 'Sales ($)'})
            fig1.update_traces(marker_color=['#667eea', '#764ba2', '#f093fb', '#4caf50'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            holiday_comp = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
            fig2 = px.bar(x=['Non-Holiday', 'Holiday'], y=holiday_comp.values,
                         title='Holiday vs Non-Holiday Sales',
                         labels={'x': '', 'y': 'Average Sales ($)'})
            fig2.update_traces(marker_color=['#667eea', '#f5576c'])
            st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🛒 Walmart Sales Prediction | Built with Streamlit & Machine Learning</p>
        <p>© 2024 Data Science Project</p>
    </div>
""", unsafe_allow_html=True)
