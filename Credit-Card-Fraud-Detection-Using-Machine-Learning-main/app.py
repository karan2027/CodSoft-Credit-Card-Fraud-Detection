import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ml_engine
import time
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="GuardianEye | Credit Card Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Visual Polish
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Background & Cards */
    .stApp {
        background-color: #0a0b10;
        color: #e2e8f0;
    }
    
    /* Header bar styling to blend with dark mode */
    header[data-testid="stHeader"] {
        background-color: #0a0b10 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0d0e16;
        border-right: 1px solid #1f2937;
    }
    
    /* Sidebar text color overrides for visibility */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }

    /* File Uploader styling for dark mode visibility */
    [data-testid="stFileUploader"] section {
        background-color: #111827 !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Neon Gradient Text for Header */
    .gradient-text {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .gradient-subtext {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Style */
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 242, 254, 0.3);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .metric-desc {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    /* Alert cards */
    .alert-card {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 12px;
        padding: 1rem;
        color: #fca5a5;
        margin-bottom: 1.5rem;
    }
    
    /* Success cards */
    .success-card {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 1rem;
        color: #6ee7b7;
        margin-bottom: 1.5rem;
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #0a0b10;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 242, 254, 0.4);
        color: #0a0b10;
    }
    
    /* Custom divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {}
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'bal_strategy' not in st.session_state:
    st.session_state.bal_strategy = 'undersample'

# ----------------- SIDEBAR CONTROLS & DATA LOAD -----------------
with st.sidebar:
    st.image("Credit Card Fraud Detection Using Machine Learning/Presentation/logo.PNG" 
             if os.path.exists("Credit Card Fraud Detection Using Machine Learning/Presentation/logo.PNG") 
             else "https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=70)
    st.markdown("### GuardianEye ML Engine")
    st.markdown("An AI-driven environment for evaluating fraud detection algorithms.")
    
    st.markdown("---")
    st.markdown("### 📥 Data Source")
    
    data_option = st.radio(
        "Choose Dataset Source:",
        ["Synthetic Data (Instant & Pre-configured)", "Upload Custom CSV", "Local creditcard.csv"]
    )
    
    load_trigger = False
    
    if data_option == "Synthetic Data (Instant & Pre-configured)":
        n_samples = st.slider("Samples to generate:", 5000, 50000, 20000, step=5000)
        fraud_pct = st.slider("Fraud Rate (%):", 0.05, 1.0, 0.17, step=0.05)
        if st.button("Generate & Load Data"):
            with st.spinner("Generating synthetic transactions..."):
                st.session_state.df = ml_engine.generate_synthetic_data(n_samples=n_samples, fraud_rate=fraud_pct/100.0)
                load_trigger = True
                
    elif data_option == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
        if uploaded_file is not None:
            if st.button("Load Uploaded Dataset"):
                with st.spinner("Reading uploaded CSV..."):
                    st.session_state.df = pd.read_csv(uploaded_file)
                    load_trigger = True
                    
    elif data_option == "Local creditcard.csv":
        if os.path.exists("creditcard.csv"):
            if st.button("Load Local Dataset"):
                with st.spinner("Reading local creditcard.csv..."):
                    st.session_state.df = pd.read_csv("creditcard.csv")
                    load_trigger = True
        else:
            st.markdown("""
            <div class='alert-card'>
                ⚠️ <b>File not found!</b><br>
                Please place <code>creditcard.csv</code> in the project directory or choose another option.
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Attempt Download (Kaggle Subset)"):
                with st.spinner("Downloading dataset subset from GitHub mirror..."):
                    success = ml_engine.try_download_subset()
                    if success:
                        st.success("Download complete! Click 'Load Local Dataset' to load.")
                    else:
                        st.error("Download failed. Please upload custom CSV or generate synthetic data.")

    # Reset trained models if data changes
    if load_trigger:
        st.session_state.models_trained = {}
        st.session_state.X_train = None
        st.success(f"Dataset Loaded Successfully! ({len(st.session_state.df):,} rows)")

# ----------------- APP CONTENT / TABS -----------------
st.markdown('<div class="gradient-text">GuardianEye 🛡️</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subtext">Real-Time Credit Card Fraud Detection and Machine Learning Dashboard</div>', unsafe_allow_html=True)

if st.session_state.df is None:
    # Landing page state before dataset loading
    st.info("👈 Please select a data source in the sidebar to load the transactions and start the dashboard.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Why Fraud Detection Matters
        Each year, credit card fraud results in **billions of dollars in losses** globally. Detecting fraud is particularly challenging because:
        1. **Extreme Imbalance**: Less than 0.2% of transactions are fraudulent. Standard ML models can achieve 99.8% accuracy simply by predicting "not fraud" every time.
        2. **PCA Anonymization**: Real transaction features (such as cardholder names, location, merchant) are hidden. Typically, only PCA components (V1-V28), time, and amount are available.
        3. **Fast Evasion**: Fraudsters constantly adapt their tactics, requiring fast, adaptive modeling.
        """)
    with col2:
        # Show an image using the generate_image tool or a public security graphic
        st.image("https://images.unsplash.com/photo-1563013544-824ae1d704d3?auto=format&fit=crop&q=80&w=800", caption="Secure Transactions Analytics")
        
else:
    # Sidebar navigation tabs
    menu = st.tabs(["🏠 Overview & EDA", "🧠 ML Training Room", "🔍 Single Transaction Detector", "📂 Batch Predictor"])
    
    df = st.session_state.df
    
    # ----------------- TAB 1: OVERVIEW & EDA -----------------
    with menu[0]:
        st.markdown("### 📊 Dataset Overview & Exploratory Data Analysis")
        
        # High Level KPIs
        total_tx = len(df)
        fraud_tx = len(df[df['Class'] == 1])
        legit_tx = len(df[df['Class'] == 0])
        fraud_rate = (fraud_tx / total_tx) * 100
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Total Transactions</div>
                <div class='metric-value'>{total_tx:,}</div>
                <div class='metric-desc'>Dataset size</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Legitimate</div>
                <div class='metric-value' style='color:#10b981;'>{legit_tx:,}</div>
                <div class='metric-desc'>Normal transactions</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Fraudulent</div>
                <div class='metric-value' style='color:#ef4444;'>{fraud_tx:,}</div>
                <div class='metric-desc'>Suspicious activities flagged</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Fraud Rate</div>
                <div class='metric-value' style='color:#f59e0b;'>{fraud_rate:.3f}%</div>
                <div class='metric-desc'>Class Imbalance ratio</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Plots layout
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            st.markdown("#### ⚖️ Class Distribution (Highly Imbalanced)")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0a0b10')
            ax.set_facecolor('#111827')
            
            sns.barplot(x=['Legit (0)', 'Fraud (1)'], y=[legit_tx, fraud_tx], palette=['#10b981', '#ef4444'], ax=ax)
            ax.set_yscale('log') # Log scale to make fraud visible
            ax.set_ylabel("Count (Log Scale)", color="#94a3b8")
            ax.tick_params(colors='#94a3b8')
            ax.spines['bottom'].set_color('#1f2937')
            ax.spines['left'].set_color('#1f2937')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
        with p_col2:
            st.markdown("#### 💸 Transaction Amount Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0a0b10')
            ax.set_facecolor('#111827')
            
            sns.histplot(data=df, x='Amount', hue='Class', bins=30, kde=True, palette=['#10b981', '#ef4444'], element='step', ax=ax)
            ax.set_xlim(0, 1000) # Zoom into typical amounts
            ax.set_xlabel("Amount ($)", color="#94a3b8")
            ax.set_ylabel("Frequency", color="#94a3b8")
            ax.tick_params(colors='#94a3b8')
            ax.spines['bottom'].set_color('#1f2937')
            ax.spines['left'].set_color('#1f2937')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # PCA Feature Inspector
        st.markdown("#### 🔍 PCA Components Discriminant Inspector")
        st.markdown("Choose a PCA component (V1-V28) to visualize how it separates Fraudulent and Legitimate transactions.")
        
        selected_v = st.selectbox("Select Feature Component:", [f'V{i}' for i in range(1, 29)])
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor('#0a0b10')
        ax.set_facecolor('#111827')
        
        sns.kdeplot(data=df[df['Class'] == 0], x=selected_v, fill=True, color='#10b981', label='Legit', ax=ax)
        sns.kdeplot(data=df[df['Class'] == 1], x=selected_v, fill=True, color='#ef4444', label='Fraud', ax=ax)
        
        ax.set_title(f"Density Distribution of {selected_v} by Class", color="white")
        ax.set_xlabel(selected_v, color="#94a3b8")
        ax.set_ylabel("Density", color="#94a3b8")
        ax.tick_params(colors='#94a3b8')
        ax.legend(facecolor='#1f2937', edgecolor='none', labelcolor='white')
        ax.spines['bottom'].set_color('#1f2937')
        ax.spines['left'].set_color('#1f2937')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        
    # ----------------- TAB 2: MODEL TRAINING ROOM -----------------
    with menu[1]:
        st.markdown("### 🧠 Machine Learning Model Training & Comparison Room")
        
        st.markdown("""
        To build a model, we split the data into **80% training** and **20% testing**. 
        Because of the severe imbalance, we must preprocess the training set using a balancing strategy:
        - **Random Undersampling**: Randomly samples legit transactions to equal fraud transactions (balanced ratio 1:1). Match the behavior in the user's project proposal.
        - **Simple Oversampling**: Over-samples the fraud class to match the legit transaction count.
        """)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            strategy = st.selectbox(
                "Choose Class Balancing Strategy:",
                ["Undersampling (Target Balanced: ~1,000 rows)", "Oversampling (Target Balanced: ~40,000 rows)"]
            )
            strategy_code = 'undersample' if "Undersampling" in strategy else 'oversample'
            
        with col_t2:
            models_to_train = st.multiselect(
                "Select Models to Train & Compare:",
                ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Random Forest"],
                default=["Logistic Regression", "Decision Tree"]
            )
            
        if st.button("🚀 Train Selected Models"):
            if not models_to_train:
                st.warning("Please select at least one machine learning model to train.")
            else:
                # Prepare features and target
                X = df.drop(columns='Class')
                y = df['Class']
                
                # Split train/test (stratify keeps target class distribution equal in splits)
                X_train_raw, X_test, y_train_raw, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                # Balance only the training set! Evaluation must be on the raw, imbalanced test set.
                with st.spinner("Balancing training set and training models..."):
                    X_train, y_train = ml_engine.balance_dataset(X_train_raw, y_train_raw, strategy=strategy_code)
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.bal_strategy = strategy_code
                    
                    st.session_state.models_trained = {}
                    
                    for model_name in models_to_train:
                        t0 = time.time()
                        metrics = ml_engine.train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test)
                        metrics["time_taken"] = time.time() - t0
                        st.session_state.models_trained[model_name] = metrics
                        
                st.success("All models trained successfully!")
                
        # Show Results if models are trained
        if st.session_state.models_trained:
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown("#### 📊 Model Performance Comparison")
            
            # Format results in table
            results_data = []
            for name, metrics in st.session_state.models_trained.items():
                results_data.append({
                    "Algorithm": name,
                    "Train Accuracy": f"{metrics['train_accuracy']*100:.2f}%",
                    "Test Accuracy": f"{metrics['test_accuracy']*100:.2f}%",
                    "Precision": f"{metrics['precision']*100:.2f}%",
                    "Recall (TPR)": f"{metrics['recall']*100:.2f}%",
                    "F1 Score": f"{metrics['f1_score']*100:.2f}%",
                    "AUC-ROC": f"{metrics['auc']:.4f}",
                    "Train Time (s)": f"{metrics['time_taken']:.3f}s"
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Metrics chart
            chart_df = []
            for name, metrics in st.session_state.models_trained.items():
                for metric_name in ["precision", "recall", "f1_score", "auc"]:
                    chart_df.append({
                        "Model": name,
                        "Metric": metric_name.upper(),
                        "Value": metrics[metric_name]
                    })
            chart_df = pd.DataFrame(chart_df)
            
            # Matplotlib metrics comparison bar plot
            fig, ax = plt.subplots(figsize=(10, 4.5))
            fig.patch.set_facecolor('#0a0b10')
            ax.set_facecolor('#111827')
            
            sns.barplot(data=chart_df, x='Metric', y='Value', hue='Model', palette='viridis', ax=ax)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score", color="#94a3b8")
            ax.tick_params(colors='#94a3b8')
            ax.legend(facecolor='#1f2937', edgecolor='none', labelcolor='white')
            ax.spines['bottom'].set_color('#1f2937')
            ax.spines['left'].set_color('#1f2937')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
            # ROC Curves & Confusion Matrix
            col_roc, col_cm = st.columns(2)
            
            with col_roc:
                st.markdown("#### 📈 ROC Curves")
                fig, ax = plt.subplots(figsize=(6, 5))
                fig.patch.set_facecolor('#0a0b10')
                ax.set_facecolor('#111827')
                
                for name, metrics in st.session_state.models_trained.items():
                    ax.plot(metrics["fpr"], metrics["tpr"], label=f"{name} (AUC = {metrics['auc']:.3f})")
                
                ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
                ax.set_xlabel("False Positive Rate", color="#94a3b8")
                ax.set_ylabel("True Positive Rate (Recall)", color="#94a3b8")
                ax.tick_params(colors='#94a3b8')
                ax.legend(facecolor='#1f2937', edgecolor='none', labelcolor='white')
                ax.spines['bottom'].set_color('#1f2937')
                ax.spines['left'].set_color('#1f2937')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
                
            with col_cm:
                st.markdown("#### 🎛️ Confusion Matrix")
                selected_model_cm = st.selectbox(
                    "Select Model for Confusion Matrix:",
                    list(st.session_state.models_trained.keys())
                )
                
                cm = np.array(st.session_state.models_trained[selected_model_cm]["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(5, 4.5))
                fig.patch.set_facecolor('#0a0b10')
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
                
                ax.set_title(f"Confusion Matrix - {selected_model_cm}", color="white")
                ax.set_xlabel("Predicted Class", color="#94a3b8")
                ax.set_ylabel("True Class", color="#94a3b8")
                ax.tick_params(colors='#94a3b8')
                st.pyplot(fig)
                
    # ----------------- TAB 3: SINGLE TRANSACTION DETECTOR -----------------
    with menu[2]:
        st.markdown("### 🔍 Single Transaction Real-Time Predictor")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train at least one machine learning model in the 'ML Training Room' tab first.")
        else:
            st.markdown("Use this tab to input details of a single transaction and test if the trained models predict it as fraudulent.")
            
            # Preset templates
            col_pre1, col_pre2 = st.columns(2)
            
            # Draw sample from dataset to pre-populate inputs
            if 'preset_inputs' not in st.session_state:
                st.session_state.preset_inputs = {}
                
            with col_pre1:
                if st.button("🟢 Load Legit Transaction Template"):
                    legit_sample = df[df['Class'] == 0].sample(1).iloc[0]
                    st.session_state.preset_inputs = legit_sample.to_dict()
                    st.success("Loaded legit transaction data features into sliders!")
                    
            with col_pre2:
                if st.button("🔴 Load Fraudulent Transaction Template"):
                    fraud_sample = df[df['Class'] == 1].sample(1).iloc[0]
                    st.session_state.preset_inputs = fraud_sample.to_dict()
                    st.success("Loaded fraud transaction data features into sliders!")
                    
            st.markdown("---")
            
            # Columns of inputs
            inp_col1, inp_col2, inp_col3 = st.columns(3)
            
            # Default values
            presets = st.session_state.preset_inputs
            
            with inp_col1:
                time_val = st.number_input("Time (seconds elapsed):", min_value=0.0, max_value=200000.0, 
                                           value=float(presets.get('Time', 0.0)))
                amount_val = st.number_input("Amount ($):", min_value=0.0, max_value=30000.0, 
                                             value=float(presets.get('Amount', 10.0)))
                v1 = st.slider("V1 (PCA component):", -30.0, 30.0, float(presets.get('V1', 0.0)))
                v2 = st.slider("V2 (PCA component):", -30.0, 30.0, float(presets.get('V2', 0.0)))
                v3 = st.slider("V3 (PCA component):", -30.0, 30.0, float(presets.get('V3', 0.0)))
                v4 = st.slider("V4 (PCA component):", -30.0, 30.0, float(presets.get('V4', 0.0)))
                v5 = st.slider("V5 (PCA component):", -30.0, 30.0, float(presets.get('V5', 0.0)))
                v6 = st.slider("V6 (PCA component):", -30.0, 30.0, float(presets.get('V6', 0.0)))
                v7 = st.slider("V7 (PCA component):", -30.0, 30.0, float(presets.get('V7', 0.0)))
                v8 = st.slider("V8 (PCA component):", -30.0, 30.0, float(presets.get('V8', 0.0)))
                v9 = st.slider("V9 (PCA component):", -30.0, 30.0, float(presets.get('V9', 0.0)))
                v10 = st.slider("V10 (PCA component):", -30.0, 30.0, float(presets.get('V10', 0.0)))
                
            with inp_col2:
                v11 = st.slider("V11 (PCA component):", -30.0, 30.0, float(presets.get('V11', 0.0)))
                v12 = st.slider("V12 (PCA component):", -30.0, 30.0, float(presets.get('V12', 0.0)))
                v13 = st.slider("V13 (PCA component):", -30.0, 30.0, float(presets.get('V13', 0.0)))
                v14 = st.slider("V14 (PCA component):", -30.0, 30.0, float(presets.get('V14', 0.0)))
                v15 = st.slider("V15 (PCA component):", -30.0, 30.0, float(presets.get('V15', 0.0)))
                v16 = st.slider("V16 (PCA component):", -30.0, 30.0, float(presets.get('V16', 0.0)))
                v17 = st.slider("V17 (PCA component):", -30.0, 30.0, float(presets.get('V17', 0.0)))
                v18 = st.slider("V18 (PCA component):", -30.0, 30.0, float(presets.get('V18', 0.0)))
                v19 = st.slider("V19 (PCA component):", -30.0, 30.0, float(presets.get('V19', 0.0)))
                
            with inp_col3:
                v20 = st.slider("V20 (PCA component):", -30.0, 30.0, float(presets.get('V20', 0.0)))
                v21 = st.slider("V21 (PCA component):", -30.0, 30.0, float(presets.get('V21', 0.0)))
                v22 = st.slider("V22 (PCA component):", -30.0, 30.0, float(presets.get('V22', 0.0)))
                v23 = st.slider("V23 (PCA component):", -30.0, 30.0, float(presets.get('V23', 0.0)))
                v24 = st.slider("V24 (PCA component):", -30.0, 30.0, float(presets.get('V24', 0.0)))
                v25 = st.slider("V25 (PCA component):", -30.0, 30.0, float(presets.get('V25', 0.0)))
                v26 = st.slider("V26 (PCA component):", -30.0, 30.0, float(presets.get('V26', 0.0)))
                v27 = st.slider("V27 (PCA component):", -30.0, 30.0, float(presets.get('V27', 0.0)))
                v28 = st.slider("V28 (PCA component):", -30.0, 30.0, float(presets.get('V28', 0.0)))
                
            st.markdown("---")
            
            # Predict Button
            if st.button("🔮 Analyze Transaction"):
                input_data = pd.DataFrame([{
                    "Time": time_val,
                    "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, 
                    "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
                    "V11": v11, "V12": v12, "V13": v13, "V14": v14, "V15": v15,
                    "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20,
                    "V21": v21, "V22": v22, "V23": v23, "V24": v24, "V25": v25,
                    "V26": v26, "V27": v27, "V28": v28,
                    "Amount": amount_val
                }])
                
                # Show result for each trained model
                results_col1, results_col2 = st.columns(2)
                
                with results_col1:
                    st.markdown("#### 🚨 Model Predictions")
                    for name, trained in st.session_state.models_trained.items():
                        clf = trained["model"]
                        pred = clf.predict(input_data)[0]
                        
                        if hasattr(clf, "predict_proba"):
                            prob = clf.predict_proba(input_data)[0][1] * 100
                            prob_str = f"({prob:.1f}% fraud probability)"
                        else:
                            prob_str = ""
                            
                        if pred == 1:
                            st.markdown(f"**{name}**: 🔴 **FRAUD DETECTED** {prob_str}")
                        else:
                            st.markdown(f"**{name}**: 🟢 **LEGITIMATE** {prob_str}")
                            
                with results_col2:
                    st.markdown("#### 🛡️ Transaction Analysis Verdict")
                    # Aggregate verdict
                    fraud_votes = 0
                    total_votes = len(st.session_state.models_trained)
                    
                    for name, trained in st.session_state.models_trained.items():
                        clf = trained["model"]
                        if clf.predict(input_data)[0] == 1:
                            fraud_votes += 1
                            
                    ratio = fraud_votes / total_votes
                    if ratio >= 0.5:
                        st.markdown(f"""
                        <div class='alert-card' style='text-align: center;'>
                            <h2 style='color: #ef4444; margin:0;'>⚠️ SUSPICIOUS</h2>
                            <p style='margin: 10px 0 0 0; font-size:1.1rem;'>
                                Flagged as <b>FRAUD</b> by {fraud_votes} of {total_votes} active models.<br>
                                <i>Recommend immediate verification or temporary block of card.</i>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='success-card' style='text-align: center;'>
                            <h2 style='color: #10b981; margin:0;'>✅ SECURE</h2>
                            <p style='margin: 10px 0 0 0; font-size:1.1rem;'>
                                Evaluated as <b>SAFE/LEGITIMATE</b> by {total_votes - fraud_votes} of {total_votes} models.<br>
                                <i>Safe transaction threshold met.</i>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

    # ----------------- TAB 4: BATCH PREDICTOR -----------------
    with menu[3]:
        st.markdown("### 📂 Batch Credit Card Fraud Predictor")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train at least one machine learning model in the 'ML Training Room' tab first.")
        else:
            st.markdown("Upload a CSV file containing transactions (matching the schema of Time, V1-V28, Amount) to batch evaluate them.")
            
            # Let the user download a sample batch file
            sample_df = df.sample(10, random_state=42).drop(columns='Class')
            st.download_button(
                "📥 Download Sample Test Batch CSV",
                sample_df.to_csv(index=False),
                "test_batch_transactions.csv",
                "text/csv"
            )
            
            st.markdown("---")
            
            batch_file = st.file_uploader("Upload Batch CSV for Analysis:", type=["csv"])
            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                
                # Check column match
                required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                missing_cols = [c for c in required_cols if c not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns in CSV: {missing_cols}")
                else:
                    st.success(f"Batch file uploaded successfully ({len(batch_df)} rows).")
                    
                    # Choose model
                    selected_model = st.selectbox(
                        "Choose Classifier Model for Batch Evaluation:",
                        list(st.session_state.models_trained.keys())
                    )
                    
                    if st.button("🔎 Run Batch Prediction"):
                        clf = st.session_state.models_trained[selected_model]["model"]
                        
                        with st.spinner("Processing batch predictions..."):
                            preds = clf.predict(batch_df[required_cols])
                            
                            if hasattr(clf, "predict_proba"):
                                probs = clf.predict_proba(batch_df[required_cols])[:, 1]
                            else:
                                probs = np.zeros(len(preds))
                                
                            result_df = batch_df.copy()
                            result_df['Predicted_Class'] = preds
                            result_df['Fraud_Probability (%)'] = np.round(probs * 100, 2)
                            result_df['Prediction'] = result_df['Predicted_Class'].apply(lambda x: "🔴 FRAUD" if x == 1 else "🟢 LEGIT")
                            
                        # Show summary
                        flagged_count = int(np.sum(preds))
                        st.markdown(f"#### Summary: Flagged **{flagged_count}** of **{len(batch_df)}** transactions as suspicious.")
                        
                        # Style table highlight fraud
                        st.dataframe(result_df[['Time', 'Amount', 'Prediction', 'Fraud_Probability (%)'] + [f'V{i}' for i in range(1, 5)]], use_container_width=True)
                        
                        # Download output
                        st.download_button(
                            "📥 Download Full Prediction Results CSV",
                            result_df.to_csv(index=False),
                            "batch_fraud_predictions.csv",
                            "text/csv"
                        )
