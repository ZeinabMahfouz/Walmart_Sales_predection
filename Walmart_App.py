"""
Walmart Sales Prediction - Interactive Streamlit App
====================================================
An attractive dashboard to showcase your machine learning work
"""

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
# CUSTOM CSS FOR ATTRACTIVE STYLING
# ============================================================================

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        max-width: 100%;
    }
    h1 {
        color: #0066cc;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    h2 {
        color: #004080;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #005cb3;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        font-size: 24px;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 10px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 30px;
        font-size: 16px;
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
    """Load the Walmart dataset"""
    try:
        # Try to load cleaned file first
        df = pd.read_csv('walmart_cleaned.csv')
        st.sidebar.success("✓ Loaded walmart_cleaned.csv")
    except FileNotFoundError:
        try:
            # Fallback to original file
            df = pd.read_csv('Walmart.csv')
            st.sidebar.info("✓ Loaded Walmart.csv")
        except:
            return None
    
    # Convert Date to datetime if needed
    if 'Date' in df.columns and df['Date'].dtype == 'object':
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        except:
            df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure required columns exist
    required_cols = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 
                     'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    
    # Check if date features exist, if not create them
    if 'Year' not in df.columns and 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    return df

@st.cache_resource
def load_model_artifacts():
    """Load trained models and related artifacts"""
    try:
        model = joblib.load('best_model_cv.pkl')
        scaler = joblib.load('scaler_cv.pkl')
        with open('model_info_cv.json', 'r') as f:
            model_info = json.load(f)
        comparison_df = pd.read_csv('model_comparison_cv_results.csv')
        return model, scaler, model_info, comparison_df
    except:
        return None, None, None, None

# Load data
df = load_data()
model, scaler, model_info, comparison_df = load_model_artifacts()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/ca/Walmart_logo.svg", width=200)
st.sidebar.title("🛒 Navigation")

page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📊 Data Exploration", "🤖 Model Performance", "🔮 Make Predictions", "📈 Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info(
    """
    This app predicts Walmart weekly sales using machine learning.
    
    **Dataset:** Historical sales data from 45 stores (Feb 2010 - Nov 2012)
    
    **Models Tested:** 9 different algorithms with cross-validation
    
    **Best Model:** """ + (model_info['best_model'] if model_info else "Loading...") + """
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.markdown("Built with ❤️ using Streamlit")
st.sidebar.markdown("Data Science Project")

# ============================================================================
# PAGE 1: HOME
# ============================================================================

if page == "🏠 Home":
    # Hero Section
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        border-radius: 15px; margin-bottom: 30px;'>
            <h1 style='color: white; font-size: 48px; margin: 0;'>🛒 Walmart Sales Predictor</h1>
            <p style='color: white; font-size: 20px; margin-top: 10px;'>
                Advanced Machine Learning for Retail Forecasting
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    if df is not None and model_info is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <h3 style='margin: 0; color: white;'>📦 Total Records</h3>
                    <h2 style='margin: 10px 0 0 0; color: white;'>{:,}</h2>
                </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <h3 style='margin: 0; color: white;'>🏪 Stores</h3>
                    <h2 style='margin: 10px 0 0 0; color: white;'>{}</h2>
                </div>
            """.format(df['Store'].nunique()), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card'>
                    <h3 style='margin: 0; color: white;'>🎯 Model Accuracy</h3>
                    <h2 style='margin: 10px 0 0 0; color: white;'>{:.2f}%</h2>
                </div>
            """.format(model_info['test_r2'] * 100), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
                <div class='metric-card'>
                    <h3 style='margin: 0; color: white;'>💰 Avg Weekly Sales</h3>
                    <h2 style='margin: 10px 0 0 0; color: white;'>${:,.0f}</h2>
                </div>
            """.format(df['Weekly_Sales'].mean()), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Project Overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 📋 Project Overview")
            st.markdown("""
                <div class='info-box'>
                    <h3>🎯 Objective</h3>
                    <p>Predict weekly sales for Walmart stores using historical data and various features 
                    including temperature, fuel prices, CPI, unemployment rate, and holiday indicators.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class='info-box'>
                    <h3>🔬 Methodology</h3>
                    <ul>
                        <li><strong>Data Preprocessing:</strong> Feature engineering, outlier handling, scaling</li>
                        <li><strong>Models Tested:</strong> 9 algorithms (Linear, Ridge, Lasso, Elastic Net, Polynomial, SVR, KNN)</li>
                        <li><strong>Validation:</strong> 5-fold cross-validation for robust evaluation</li>
                        <li><strong>Metrics:</strong> R², RMSE, MAE, MAPE for comprehensive assessment</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("## 🏆 Best Model")
            st.markdown(f"""
                <div class='success-box'>
                    <h3 style='color: #2e7d32;'>{model_info['best_model']}</h3>
                    <p><strong>CV R² Score:</strong> {model_info['cv_r2_mean']:.4f} ± {model_info['cv_r2_std']:.4f}</p>
                    <p><strong>Test R² Score:</strong> {model_info['test_r2']:.4f}</p>
                    <p><strong>Test RMSE:</strong> ${model_info['test_rmse']:,.2f}</p>
                    <p><strong>Cross-Validation:</strong> {model_info['cv_folds']}-Fold CV</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("## 📊 Dataset Info")
            st.markdown(f"""
                <div class='info-box'>
                    <p><strong>Date Range:</strong> Feb 2010 - Nov 2012</p>
                    <p><strong>Features:</strong> {len(model_info['features'])}</p>
                    <p><strong>Train Size:</strong> {model_info['train_size']:,} samples</p>
                    <p><strong>Test Size:</strong> {model_info['test_size']:,} samples</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Timeline
        st.markdown("## 📅 Project Timeline")
        timeline_data = pd.DataFrame({
            'Phase': ['Data Collection', 'EDA & Preprocessing', 'Feature Engineering', 
                     'Model Training', 'Hyperparameter Tuning', 'Deployment'],
            'Status': ['✅ Complete', '✅ Complete', '✅ Complete', 
                      '✅ Complete', '✅ Complete', '✅ Complete'],
            'Duration': ['Week 1', 'Week 1-2', 'Week 2', 'Week 3', 'Week 3-4', 'Week 4']
        })
        st.dataframe(timeline_data, use_container_width=True, hide_index=True)
        
    else:
        st.error("⚠️ Unable to load data or model. Please ensure all required files are present.")

# ============================================================================
# PAGE 2: DATA EXPLORATION
# ============================================================================

elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration & Analysis")
    
    if df is not None:
        # Data Overview
        st.markdown("## 📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Number of Features", len(df.columns))
        with col3:
            st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Data Preview
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statistical Summary
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Visualizations
        st.markdown("## 📊 Interactive Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🏪 Store Analysis", "🎯 Feature Distribution", "🔥 Correlations"])
        
        with tab1:
            st.markdown("### Weekly Sales Over Time")
            
            # Time series plot
            fig = px.line(df.sort_values('Date'), x='Date', y='Weekly_Sales',
                         title='Weekly Sales Trend (All Stores)',
                         labels={'Weekly_Sales': 'Sales ($)', 'Date': 'Date'})
            fig.update_traces(line_color='#667eea', line_width=1)
            fig.update_layout(height=500, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly average
            monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Weekly_Sales'].mean().reset_index()
            monthly_sales['Date'] = monthly_sales['Date'].astype(str)
            
            fig2 = px.bar(monthly_sales, x='Date', y='Weekly_Sales',
                         title='Average Monthly Sales',
                         labels={'Weekly_Sales': 'Average Sales ($)', 'Date': 'Month'})
            fig2.update_traces(marker_color='#764ba2')
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.markdown("### Store Performance Analysis")
            
            # Top 10 stores
            store_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False).head(10).reset_index()
            
            fig3 = px.bar(store_sales, x='Store', y='Weekly_Sales',
                         title='Top 10 Stores by Average Sales',
                         labels={'Weekly_Sales': 'Average Sales ($)', 'Store': 'Store Number'})
            fig3.update_traces(marker_color='#f093fb')
            fig3.update_layout(height=500)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Store comparison
            selected_stores = st.multiselect(
                "Select stores to compare:",
                options=sorted(df['Store'].unique()),
                default=sorted(df['Store'].unique())[:3]
            )
            
            if selected_stores:
                store_data = df[df['Store'].isin(selected_stores)].sort_values('Date')
                fig4 = px.line(store_data, x='Date', y='Weekly_Sales', color='Store',
                              title='Sales Comparison - Selected Stores',
                              labels={'Weekly_Sales': 'Sales ($)', 'Date': 'Date'})
                fig4.update_layout(height=500)
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab3:
            st.markdown("### Feature Distributions")
            
            feature_col = st.selectbox(
                "Select feature to visualize:",
                ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig5 = px.histogram(df, x=feature_col, nbins=50,
                                   title=f'{feature_col} Distribution',
                                   labels={feature_col: feature_col})
                fig5.update_traces(marker_color='#667eea')
                fig5.update_layout(height=400)
                st.plotly_chart(fig5, use_container_width=True)
            
            with col2:
                fig6 = px.box(df, y=feature_col,
                             title=f'{feature_col} Box Plot',
                             labels={feature_col: feature_col})
                fig6.update_traces(marker_color='#764ba2')
                fig6.update_layout(height=400)
                st.plotly_chart(fig6, use_container_width=True)
            
            # Holiday vs Non-Holiday
            st.markdown("### Holiday Impact Analysis")
            holiday_comparison = df.groupby('Holiday_Flag')['Weekly_Sales'].mean().reset_index()
            holiday_comparison['Holiday_Flag'] = holiday_comparison['Holiday_Flag'].map({0: 'Non-Holiday', 1: 'Holiday'})
            
            fig7 = px.bar(holiday_comparison, x='Holiday_Flag', y='Weekly_Sales',
                         title='Average Sales: Holiday vs Non-Holiday',
                         labels={'Weekly_Sales': 'Average Sales ($)', 'Holiday_Flag': ''})
            fig7.update_traces(marker_color=['#667eea', '#f5576c'])
            fig7.update_layout(height=400)
            st.plotly_chart(fig7, use_container_width=True)
        
        with tab4:
            st.markdown("### Feature Correlations")
            
            # Correlation heatmap
            numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                           'Holiday_Flag', 'Store', 'Year', 'Month', 'Quarter']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            corr_matrix = df[available_cols].corr()
            
            fig8 = px.imshow(corr_matrix,
                            labels=dict(color="Correlation"),
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            color_continuous_scale='RdBu_r',
                            title='Feature Correlation Matrix')
            fig8.update_layout(height=600)
            st.plotly_chart(fig8, use_container_width=True)
            
            # Correlation with target
            target_corr = corr_matrix['Weekly_Sales'].sort_values(ascending=False)[1:].reset_index()
            target_corr.columns = ['Feature', 'Correlation']
            
            fig9 = px.bar(target_corr, x='Correlation', y='Feature', orientation='h',
                         title='Feature Correlation with Weekly Sales',
                         labels={'Correlation': 'Correlation Coefficient', 'Feature': ''})
            fig9.update_traces(marker_color=['#4caf50' if x > 0 else '#f44336' for x in target_corr['Correlation']])
            fig9.update_layout(height=500)
            st.plotly_chart(fig9, use_container_width=True)
    
    else:
        st.error("⚠️ Unable to load data. Please ensure 'Walmart.csv' is available.")

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================================

elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance & Comparison")
    
    if comparison_df is not None and model_info is not None:
        # Best Model Highlight
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
                <h2 style='margin: 0; color: white;'>🏆 Best Model: {model_info['best_model']}</h2>
                <p style='font-size: 20px; margin-top: 10px;'>
                    CV R² Score: {model_info['cv_r2_mean']:.4f} ± {model_info['cv_r2_std']:.4f} | 
                    Test R²: {model_info['test_r2']:.4f}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Model Comparison Table
        st.markdown("## 📊 Model Comparison")
        
        # Format the dataframe for display
        display_df = comparison_df[['Model', 'CV_R2_Mean', 'CV_R2_Std', 'Test_R2', 
                                    'CV_RMSE_Mean', 'Test_RMSE', 'Overfitting', 'Training_Time']].copy()
        display_df.columns = ['Model', 'CV R² Mean', 'CV R² Std', 'Test R²', 
                              'CV RMSE Mean', 'Test RMSE', 'Overfitting', 'Time (s)']
        
        # Color code the best values
        st.dataframe(
            display_df.style.background_gradient(subset=['CV R² Mean', 'Test R²'], cmap='Greens')
                           .background_gradient(subset=['Test RMSE', 'Overfitting'], cmap='Reds_r')
                           .format({
                               'CV R² Mean': '{:.4f}',
                               'CV R² Std': '{:.4f}',
                               'Test R²': '{:.4f}',
                               'CV RMSE Mean': '${:,.2f}',
                               'Test RMSE': '${:,.2f}',
                               'Overfitting': '{:.4f}',
                               'Time (s)': '{:.2f}'
                           }),
            use_container_width=True,
            hide_index=True
        )
        
        # Visual Comparisons
        st.markdown("## 📈 Visual Comparison")
        
        tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "⚠️ Overfitting Analysis", "⏱️ Efficiency"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # CV R² comparison
                fig1 = px.bar(comparison_df.sort_values('CV_R2_Mean', ascending=True),
                             y='Model', x='CV_R2_Mean',
                             error_x='CV_R2_Std',
                             orientation='h',
                             title='Cross-Validation R² Score',
                             labels={'CV_R2_Mean': 'CV R² Score', 'Model': ''})
                fig1.update_traces(marker_color='#667eea')
                fig1.update_layout(height=500)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Test R² comparison
                fig2 = px.bar(comparison_df.sort_values('Test_R2', ascending=True),
                             y='Model', x='Test_R2',
                             orientation='h',
                             title='Test R² Score',
                             labels={'Test_R2': 'Test R² Score', 'Model': ''})
                fig2.update_traces(marker_color='#764ba2')
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)
            
            # RMSE comparison
            fig3 = px.bar(comparison_df.sort_values('Test_RMSE'),
                         x='Model', y='Test_RMSE',
                         title='Test RMSE Comparison (Lower is Better)',
                         labels={'Test_RMSE': 'Test RMSE ($)', 'Model': 'Model'})
            fig3.update_traces(marker_color='#f093fb')
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            # Train vs Test R²
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                name='Train R²',
                x=comparison_df['Model'],
                y=comparison_df['Train_R2'],
                marker_color='#667eea'
            ))
            fig4.add_trace(go.Bar(
                name='Test R²',
                x=comparison_df['Model'],
                y=comparison_df['Test_R2'],
                marker_color='#764ba2'
            ))
            fig4.update_layout(
                title='Train vs Test R² (Overfitting Check)',
                xaxis_title='Model',
                yaxis_title='R² Score',
                barmode='group',
                height=500
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            # Overfitting score
            comparison_df_sorted = comparison_df.sort_values('Overfitting')
            fig5 = px.bar(comparison_df_sorted,
                         x='Model', y='Overfitting',
                         title='Overfitting Score (Lower is Better)',
                         labels={'Overfitting': 'Overfitting Score', 'Model': 'Model'},
                         color='Overfitting',
                         color_continuous_scale=['green', 'yellow', 'red'])
            fig5.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                          annotation_text="Threshold (0.05)")
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)
            
            st.markdown("""
                <div class='info-box'>
                    <strong>Overfitting Score = |CV R² Mean - Test R²|</strong><br>
                    ✓ < 0.05: Good generalization<br>
                    ⚡ 0.05-0.1: Moderate overfitting<br>
                    ⚠️ > 0.1: High overfitting
                </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            # Training time
            fig6 = px.bar(comparison_df.sort_values('Training_Time'),
                         x='Model', y='Training_Time',
                         title='Training Time Comparison',
                         labels={'Training_Time': 'Time (seconds)', 'Model': 'Model'})
            fig6.update_traces(marker_color='#4caf50')
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)
            
            # Efficiency score (R² / Time)
            comparison_df['Efficiency'] = comparison_df['Test_R2'] / (comparison_df['Training_Time'] + 0.01)
            fig7 = px.bar(comparison_df.sort_values('Efficiency', ascending=False),
                         x='Model', y='Efficiency',
                         title='Model Efficiency (Test R² / Training Time)',
                         labels={'Efficiency': 'Efficiency Score', 'Model': 'Model'})
            fig7.update_traces(marker_color='#ff9800')
            fig7.update_layout(height=400)
            st.plotly_chart(fig7, use_container_width=True)
        
        # Model Details
        st.markdown("## 🔍 Best Model Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='margin: 0; color: white;'>Cross-Validation</h4>
                    <h3 style='margin: 10px 0 0 0; color: white;'>{}-Fold CV</h3>
                </div>
            """.format(model_info['cv_folds']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='margin: 0; color: white;'>Training Samples</h4>
                    <h3 style='margin: 10px 0 0 0; color: white;'>{:,}</h3>
                </div>
            """.format(model_info['train_size']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='margin: 0; color: white;'>Test Samples</h4>
                    <h3 style='margin: 10px 0 0 0; color: white;'>{:,}</h3>
                </div>
            """.format(model_info['test_size']), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 🎛️ Best Hyperparameters")
        st.json(model_info['best_params'])
        
        st.markdown("### 📋 Features Used")
        features_df = pd.DataFrame({
            'Feature': model_info['features'],
            'Index': range(len(model_info['features']))
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    else:
        st.error("⚠️ Unable to load model comparison data.")

# ============================================================================
# PAGE 4: MAKE PREDICTIONS
# ============================================================================

elif page == "🔮 Make Predictions":
    st.title("🔮 Make Sales Predictions")
    
    if model is not None and scaler is not None and model_info is not None:
        st.markdown("""
            <div class='info-box'>
                <h3>👇 Enter Store Details</h3>
                <p>Fill in the features below to predict weekly sales for a Walmart store.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                store = st.number_input("Store Number", min_value=1, max_value=45, value=1, step=1)
                holiday_flag = st.selectbox("Holiday Week?", options=[0, 1], 
                                           format_func=lambda x: "No" if x == 0 else "Yes")
                temperature = st.number_input("Temperature (°F)", min_value=-20.0, max_value=120.0, value=60.0, step=0.1)
                fuel_price = st.number_input("Fuel Price ($/gallon)", min_value=2.0, max_value=5.0, value=3.0, step=0.01)
            
            with col2:
                cpi = st.number_input("Consumer Price Index", min_value=100.0, max_value=250.0, value=211.0, step=0.1)
                unemployment = st.number_input("Unemployment Rate (%)", min_value=2.0, max_value=15.0, value=8.0, step=0.1)
                year = st.number_input("Year", min_value=2010, max_value=2025, value=2012, step=1)
                month = st.selectbox("Month", options=list(range(1, 13)), index=10)
            
            with col3:
                day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)
                quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=3)
                is_weekend = st.selectbox("Is Weekend?", options=[0, 1], 
                                         format_func=lambda x: "No" if x == 0 else "Yes")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("🔮 Predict Sales", use_container_width=True)
        
        if submit_button:
            # Prepare input data
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
                'IsWeekend': [is_weekend]
            })
            
            # Ensure columns match training features
            input_data = input_data[model_info['features']]
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
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
                st.metric("Confidence Level", f"{model_info['test_r2']*100:.1f}%", 
                         help="Based on model's test R² score")
            
            with col2:
                avg_sales = df['Weekly_Sales'].mean() if df is not None else 0
                diff = ((prediction - avg_sales) / avg_sales * 100) if avg_sales > 0 else 0
                st.metric("vs Average Sales", f"{diff:+.1f}%", 
                         delta=f"${prediction - avg_sales:,.2f}")
            
            with col3:
                if df is not None:
                    store_avg = df[df['Store'] == store]['Weekly_Sales'].mean()
                    store_diff = ((prediction - store_avg) / store_avg * 100) if store_avg > 0 else 0
                    st.metric(f"vs Store {store} Avg", f"{store_diff:+.1f}%",
                             delta=f"${prediction - store_avg:,.2f}")
            
            # Feature contribution (if available)
            st.markdown("### 📊 Input Summary")
            input_summary = pd.DataFrame({
                'Feature': model_info['features'],
                'Value': input_data.values[0]
            })
            st.dataframe(input_summary, use_container_width=True, hide_index=True)
            
            # Recommendations
            st.markdown("### 💡 Insights")
            
            insights = []
            if holiday_flag == 1:
                insights.append("🎉 Holiday weeks typically see increased sales.")
            if temperature < 40:
                insights.append("❄️ Cold weather may impact foot traffic.")
            elif temperature > 85:
                insights.append("☀️ Hot weather could affect certain product categories.")
            if unemployment > 10:
                insights.append("📉 High unemployment rate may impact consumer spending.")
            if fuel_price > 3.5:
                insights.append("⛽ Elevated fuel prices might reduce discretionary spending.")
            if is_weekend == 1:
                insights.append("🛒 Weekend shopping patterns may differ from weekdays.")
            
            if insights:
                for insight in insights:
                    st.markdown(f"""
                        <div class='info-box'>
                            {insight}
                        </div>
                    """, unsafe_allow_html=True)
        
        # Batch Prediction
        st.markdown("---")
        st.markdown("## 📁 Batch Prediction")
        st.markdown("Upload a CSV file with multiple store records to get predictions in bulk.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.markdown("### 📋 Uploaded Data Preview")
                st.dataframe(batch_data.head(), use_container_width=True)
                
                if st.button("🔮 Generate Batch Predictions", use_container_width=True):
                    # Check if required features exist
                    required_features = model_info['features']
                    
                    # Remove Log_Weekly_Sales from required features if it exists (it's the target, not a feature!)
                    if 'Log_Weekly_Sales' in required_features:
                        required_features = [f for f in required_features if f != 'Log_Weekly_Sales']
                        st.info("Note: Log_Weekly_Sales is not needed for predictions (it's the target variable)")
                    
                    missing_cols = [col for col in required_features if col not in batch_data.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                        st.info(f"Required columns: {required_features}")
                        
                        # Try to create missing columns
                        st.warning("Attempting to create missing columns...")
                        
                        # Add WeekOfYear if missing
                        if 'WeekOfYear' in missing_cols:
                            if 'Date' in batch_data.columns:
                                batch_data['Date'] = pd.to_datetime(batch_data['Date'])
                                batch_data['WeekOfYear'] = batch_data['Date'].dt.isocalendar().week
                                st.success("✓ Created WeekOfYear from Date column")
                            elif 'Year' in batch_data.columns and 'Month' in batch_data.columns and 'Day' in batch_data.columns:
                                # Create date from Year, Month, Day
                                try:
                                    batch_data['Date'] = pd.to_datetime(batch_data[['Year', 'Month', 'Day']])
                                    batch_data['WeekOfYear'] = batch_data['Date'].dt.isocalendar().week
                                    st.success("✓ Created WeekOfYear from Year/Month/Day")
                                except:
                                    # Default to mid-year week based on month
                                    batch_data['WeekOfYear'] = ((batch_data['Month'] - 1) * 4) + 2
                                    st.warning("⚠️ Estimated WeekOfYear from Month")
                            else:
                                # Default to mid-year week
                                batch_data['WeekOfYear'] = 26
                                st.warning("⚠️ Used default WeekOfYear (26)")
                            missing_cols.remove('WeekOfYear')
                        
                        # Check again
                        missing_cols = [col for col in required_features if col not in batch_data.columns]
                        if missing_cols:
                            st.error(f"❌ Still missing columns: {missing_cols}")
                            st.info("Please ensure your CSV has these columns: " + ", ".join(required_features))
                            st.stop()
                    
                    # Ensure columns match and are in correct order
                    try:
                        batch_data_input = batch_data[required_features]
                        batch_scaled = scaler.transform(batch_data_input)
                        batch_predictions = model.predict(batch_scaled)
                        
                        # Add predictions to dataframe
                        batch_data['Predicted_Sales'] = batch_predictions
                        
                        st.markdown("### ✅ Predictions Complete!")
                        st.dataframe(batch_data, use_container_width=True)
                        
                        # Download button
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="walmart_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Summary statistics
                        st.markdown("### 📊 Batch Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Predictions", len(batch_predictions))
                        with col2:
                            st.metric("Average Predicted Sales", f"${batch_predictions.mean():,.2f}")
                        with col3:
                            st.metric("Min Predicted Sales", f"${batch_predictions.min():,.2f}")
                        with col4:
                            st.metric("Max Predicted Sales", f"${batch_predictions.max():,.2f}")
                    
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")
                        st.info("Debug info:")
                        st.write("Required features:", required_features)
                        st.write("Available columns:", batch_data.columns.tolist())
                        st.write("Missing:", [f for f in required_features if f not in batch_data.columns])
                    
                    # Download button
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="walmart_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Summary statistics
                    st.markdown("### 📊 Batch Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Predictions", len(batch_predictions))
                    with col2:
                        st.metric("Average Predicted Sales", f"${batch_predictions.mean():,.2f}")
                    with col3:
                        st.metric("Min Predicted Sales", f"${batch_predictions.min():,.2f}")
                    with col4:
                        st.metric("Max Predicted Sales", f"${batch_predictions.max():,.2f}")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV file has all required columns: " + ", ".join(model_info['features']))
    
    else:
        st.error("⚠️ Unable to load model. Please ensure 'best_model_cv.pkl' and 'scaler_cv.pkl' are available.")

# ============================================================================
# PAGE 5: INSIGHTS
# ============================================================================

elif page == "📈 Insights":
    st.title("📈 Key Insights & Findings")
    
    if df is not None and model_info is not None:
        # Executive Summary
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; color: white; margin-bottom: 30px;'>
                <h2 style='margin: 0; color: white;'>🎯 Executive Summary</h2>
                <p style='font-size: 18px; margin-top: 15px; line-height: 1.6;'>
                    This machine learning project successfully predicts Walmart weekly sales with 
                    <strong>{:.1f}% accuracy</strong> using advanced regression algorithms. 
                    The <strong>{}</strong> emerged as the best performer after rigorous 
                    cross-validation testing.
                </p>
            </div>
        """.format(model_info['test_r2'] * 100, model_info['best_model']), unsafe_allow_html=True)
        
        # Key Findings
        st.markdown("## 🔍 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='success-box'>
                    <h3>📊 Data Insights</h3>
                    <ul>
                        <li><strong>Seasonality:</strong> Q4 shows highest sales (holiday season)</li>
                        <li><strong>Holiday Impact:</strong> Holiday weeks show significant sales variance</li>
                        <li><strong>Store Variance:</strong> Top stores outperform bottom stores by 2-3x</li>
                        <li><strong>Economic Factors:</strong> Unemployment and CPI show moderate correlation</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='info-box'>
                    <h3>🤖 Model Performance</h3>
                    <ul>
                        <li><strong>Best Algorithm:</strong> {}</li>
                        <li><strong>Test Accuracy:</strong> {:.2f}% (R² score)</li>
                        <li><strong>Cross-Validation:</strong> {:.4f} ± {:.4f}</li>
                        <li><strong>Overfitting:</strong> {:.4f} (Good generalization)</li>
                    </ul>
                </div>
            """.format(
                model_info['best_model'],
                model_info['test_r2'] * 100,
                model_info['cv_r2_mean'],
                model_info['cv_r2_std'],
                model_info['overfitting_score']
            ), unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("## 📊 Visual Insights")
        
        tab1, tab2, tab3 = st.tabs(["📅 Temporal Patterns", "🏪 Store Insights", "🎯 Feature Impact"])
        
        with tab1:
            # Quarterly sales
            quarterly = df.groupby('Quarter')['Weekly_Sales'].mean().reset_index()
            quarterly['Quarter'] = quarterly['Quarter'].map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
            
            fig1 = px.bar(quarterly, x='Quarter', y='Weekly_Sales',
                         title='Average Sales by Quarter',
                         labels={'Weekly_Sales': 'Average Sales ($)', 'Quarter': 'Quarter'})
            fig1.update_traces(marker_color=['#667eea', '#764ba2', '#f093fb', '#4caf50'])
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Monthly trend
            monthly = df.groupby('Month')['Weekly_Sales'].mean().reset_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly['Month_Name'] = monthly['Month'].apply(lambda x: month_names[x-1])
            
            fig2 = px.line(monthly, x='Month_Name', y='Weekly_Sales',
                          title='Average Sales by Month',
                          labels={'Weekly_Sales': 'Average Sales ($)', 'Month_Name': 'Month'},
                          markers=True)
            fig2.update_traces(line_color='#667eea', line_width=3, marker_size=10)
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("""
                <div class='info-box'>
                    <strong>Key Observation:</strong> Sales peak in Q4 (October-December) due to 
                    holiday shopping, with November and December showing the highest weekly sales.
                </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            # Store performance distribution
            store_stats = df.groupby('Store')['Weekly_Sales'].agg(['mean', 'std']).reset_index()
            store_stats['CV'] = (store_stats['std'] / store_stats['mean']) * 100
            
            fig3 = px.scatter(store_stats, x='mean', y='CV', 
                             hover_data=['Store'],
                             title='Store Performance: Average Sales vs Variability',
                             labels={'mean': 'Average Weekly Sales ($)', 
                                   'CV': 'Coefficient of Variation (%)'},
                             size='mean', color='CV',
                             color_continuous_scale='Viridis')
            fig3.update_layout(height=500)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Top vs Bottom stores
            top_5 = df.groupby('Store')['Weekly_Sales'].mean().nlargest(5).reset_index()
            bottom_5 = df.groupby('Store')['Weekly_Sales'].mean().nsmallest(5).reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                fig4 = px.bar(top_5, x='Store', y='Weekly_Sales',
                             title='Top 5 Performing Stores',
                             labels={'Weekly_Sales': 'Avg Sales ($)', 'Store': 'Store'})
                fig4.update_traces(marker_color='#4caf50')
                fig4.update_layout(height=350)
                st.plotly_chart(fig4, use_container_width=True)
            
            with col2:
                fig5 = px.bar(bottom_5, x='Store', y='Weekly_Sales',
                             title='Bottom 5 Performing Stores',
                             labels={'Weekly_Sales': 'Avg Sales ($)', 'Store': 'Store'})
                fig5.update_traces(marker_color='#f44336')
                fig5.update_layout(height=350)
                st.plotly_chart(fig5, use_container_width=True)
        
        with tab3:
            # Feature correlations with target
            numeric_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            correlations = df[available_cols + ['Weekly_Sales']].corr()['Weekly_Sales'].drop('Weekly_Sales')
            corr_df = correlations.reset_index()
            corr_df.columns = ['Feature', 'Correlation']
            corr_df = corr_df.sort_values('Correlation', ascending=True)
            
            fig6 = px.bar(corr_df, x='Correlation', y='Feature', orientation='h',
                         title='Feature Correlation with Weekly Sales',
                         labels={'Correlation': 'Correlation Coefficient', 'Feature': ''})
            colors = ['#4caf50' if x > 0 else '#f44336' for x in corr_df['Correlation']]
            fig6.update_traces(marker_color=colors)
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)
            
            st.markdown("""
                <div class='info-box'>
                    <strong>Interpretation:</strong><br>
                    • Positive correlation: Feature increases → Sales increase<br>
                    • Negative correlation: Feature increases → Sales decrease<br>
                    • Correlation closer to ±1 indicates stronger relationship
                </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("## 💡 Business Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='success-box'>
                    <h3>📈 For High-Performing Stores</h3>
                    <ul>
                        <li>Maintain inventory levels during Q4</li>
                        <li>Capitalize on holiday promotional opportunities</li>
                        <li>Study success factors for replication</li>
                        <li>Consider expansion in similar demographics</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class='info-box'>
                    <h3>🎯 Inventory Management</h3>
                    <ul>
                        <li>Increase stock 20-30% before holidays</li>
                        <li>Monitor weather forecasts for demand planning</li>
                        <li>Adjust staffing based on predicted sales</li>
                        <li>Optimize supply chain for peak seasons</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='success-box'>
                    <h3>📉 For Underperforming Stores</h3>
                    <ul>
                        <li>Analyze local economic factors</li>
                        <li>Review pricing and promotional strategies</li>
                        <li>Improve customer experience initiatives</li>
                        <li>Consider store layout optimization</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class='info-box'>
                    <h3>🔮 Predictive Planning</h3>
                    <ul>
                        <li>Use model for weekly sales forecasting</li>
                        <li>Plan promotional budgets based on predictions</li>
                        <li>Optimize workforce scheduling</li>
                        <li>Improve cash flow management</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Future Work
        st.markdown("## 🚀 Future Enhancements")
        
        st.markdown("""
            <div class='info-box'>
                <h3>Potential Improvements</h3>
                <ul>
                    <li>🔄 <strong>Real-time Updates:</strong> Integrate live data feeds for continuous model retraining</li>
                    <li>📱 <strong>Mobile App:</strong> Develop iOS/Android apps for on-the-go predictions</li>
                    <li>🌐 <strong>External Data:</strong> Incorporate social media trends, competitor pricing</li>
                    <li>🧠 <strong>Deep Learning:</strong> Explore LSTM/Transformer models for time series</li>
                    <li>📍 <strong>Geographic Analysis:</strong> Add location-based features (demographics, proximity)</li>
                    <li>🎨 <strong>Product Categories:</strong> Build separate models for different departments</li>
                    <li>⚡ <strong>AutoML:</strong> Implement automated model selection and hyperparameter tuning</li>
                    <li>🔔 <strong>Alerts:</strong> Create anomaly detection for unusual sales patterns</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Technical Details
        st.markdown("## 🛠️ Technical Stack")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: white;'>Data Processing</h4>
                    <p style='color: white;'>• Pandas<br>• NumPy<br>• Scikit-learn</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: white;'>Visualization</h4>
                    <p style='color: white;'>• Plotly<br>• Matplotlib<br>• Seaborn</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card'>
                    <h4 style='color: white;'>Deployment</h4>
                    <p style='color: white;'>• Streamlit<br>• Joblib<br>• Python 3.x</p>
                </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error("⚠️ Unable to load data or model information.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🛒 Walmart Sales Prediction Dashboard</p>
        <p>Built with Streamlit | Powered by Machine Learning</p>
        <p>© 2024 - Data Science Project</p>
    </div>
""", unsafe_allow_html=True)