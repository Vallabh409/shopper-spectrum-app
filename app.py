import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Product Recommendation & Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data and Models ---
# The code assumes these files are in the same directory as the app.py script.
try:
    # Load the similarity matrix for item-based collaborative filtering
    similarity_df = pd.read_csv('similarity.csv', index_col=0)
    
    # Load the product lookup table to map StockCode to Description
    product_lookup_df = pd.read_csv('product_lookup.csv')
    
    # Load the pre-trained K-Means model for customer segmentation
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
        
    # Load the scaler used for the K-Means model
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Clean the product lookup data
    product_lookup_df['Description'] = product_lookup_df['Description'].str.strip()
    product_lookup_df.drop_duplicates(subset=['Description'], inplace=True)
    product_lookup_df.set_index('Description', inplace=True)
    
except FileNotFoundError as e:
    st.error(f"Error: A required file was not found. Please ensure that 'kmeans_model.pkl', 'scaler.pkl', 'similarity.csv', and 'product_lookup.csv' are in the same directory as the Streamlit app. Details: {e}")
    st.stop()


# --- Main App Title ---
st.title("🛍️ E-commerce Analysis: Recommendation & Segmentation")
st.markdown("---")

# --- Recommendation System Approach ---
# This is a good place to explain the methodology
st.header("🎯 Product Recommendation Module")
st.markdown("This module provides product recommendations based on **Item-based Collaborative Filtering**. "
            "We use cosine similarity on the `CustomerID-StockCode` purchase history matrix to find "
            "products that are frequently bought together.")

# Function to get recommendations
def get_recommendations(product_name, n=5):
    """
    Finds the top N most similar products to the given product.
    
    Args:
        product_name (str): The name of the product to find similar items for.
        n (int): The number of recommendations to return.
        
    Returns:
        list: A list of recommended product descriptions.
    """
    try:
        # Get the StockCode from the product name
        stock_code = product_lookup_df.loc[product_name, 'StockCode']
        
        # Get the similarity scores for the product
        similar_products = similarity_df[str(stock_code)].sort_values(ascending=False)
        
        # Drop the product itself and get the top N
        similar_products = similar_products.drop(str(stock_code))
        top_similar_products = similar_products.head(n)
        
        # Get the StockCodes of the recommended products
        recommended_stock_codes = top_similar_products.index.astype(int)
        
        # Map the StockCodes back to product descriptions
        recommended_products = [
            product_lookup_df[product_lookup_df['StockCode'] == code].iloc[0]['Description']
            for code in recommended_stock_codes
        ]
        
        return recommended_products
    except KeyError:
        st.warning(f"Product '{product_name}' not found. Please try a different product.")
        return []

# Recommendation UI
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        product_name_input = st.text_input(
            "Product Name",
            placeholder="Enter a product name (e.g., JUMBO BAG RED RETROSPOT)",
            help="Please enter the full product name as it appears in the data."
        )
    
    with col2:
        st.write("") # Add some space
        st.write("") # Add some space
        get_rec_button = st.button("Get Recommendations", use_container_width=True)

if get_rec_button and product_name_input:
    st.subheader("Your Top 5 Recommendations:")
    recommendations = get_recommendations(product_name_input)
    if recommendations:
        for i, rec in enumerate(recommendations):
            st.markdown(f"**{i+1}.** {rec}")
    else:
        st.info("No recommendations found. Please try another product name.")

st.markdown("---")

# --- Customer Segmentation Module ---
# Explain the methodology
st.header("🎯 Customer Segmentation Module")
st.markdown("This module predicts a customer's segment based on their Recency, Frequency, and Monetary values.")
st.markdown("The model uses **RFM analysis** and a **K-Means clustering** model trained on these metrics.")
st.markdown("The predicted cluster labels are mapped to descriptive customer segments.")

# Function to predict the customer cluster
def predict_customer_cluster(recency, frequency, monetary):
    """
    Predicts the customer cluster based on RFM values.
    
    Args:
        recency (int): The number of days since the last purchase.
        frequency (int): The number of purchases made.
        monetary (float): The total amount spent.
        
    Returns:
        str: The predicted cluster label.
    """
    # Create a DataFrame from the user inputs
    # The columns are named to match the features used for training the model.
    input_data = pd.DataFrame([[recency, frequency, monetary]], 
                              columns=['Recency', 'Frequency', 'Monetary'])
    
    # Log transform the RFM values to normalize the skewed distribution
    input_data['Recency_log'] = np.log1p(input_data['Recency'])
    input_data['Frequency_log'] = np.log1p(input_data['Frequency'])
    input_data['Monetary_log'] = np.log1p(input_data['Monetary'])
    
    # Ensure the input data has the exact columns and order that the scaler was trained on.
    # The image shows the scaler was trained on the log-transformed features only,
    # so we will use only those for the prediction.
    input_data = input_data[['Recency_log', 'Frequency_log', 'Monetary_log']]
    
    # Scale the input data using the pre-trained scaler
    scaled_data = scaler.transform(input_data)
    
    # Predict the cluster using the pre-trained K-Means model
    cluster = kmeans_model.predict(scaled_data)[0]
    
    # Map the cluster label to a descriptive name
    cluster_mapping = {
        0: 'High-Value',
        1: 'Regular',
        2: 'Occasional',
        3: 'At-Risk'
    }
    
    return cluster_mapping.get(cluster, "Unknown Cluster")

# Segmentation UI
with st.container():
    recency = st.number_input(
        "Recency (in days)",
        min_value=0,
        value=30,
        help="Number of days since the customer's last purchase."
    )
    frequency = st.number_input(
        "Frequency (number of purchases)",
        min_value=0,
        value=5,
        help="Total number of purchases made by the customer."
    )
    monetary = st.number_input(
        "Monetary (total spend)",
        min_value=0.0,
        value=150.0,
        help="Total monetary value of the customer's purchases."
    )
    
    predict_button = st.button("Predict Cluster", use_container_width=True)

if predict_button:
    st.subheader("Predicted Customer Cluster:")
    cluster_label = predict_customer_cluster(recency, frequency, monetary)
    
    # Display the result with styling
    if cluster_label == 'High-Value':
        st.success(f"Cluster: **{cluster_label}** 🎉")
    elif cluster_label == 'At-Risk':
        st.error(f"Cluster: **{cluster_label}** ⚠️")
    elif cluster_label == 'Regular':
        st.info(f"Cluster: **{cluster_label}** 👍")
    else:
        st.write(f"Cluster: **{cluster_label}**")

st.markdown("---")
st.sidebar.markdown("# App Information")
st.sidebar.info("This app demonstrates a product recommendation and customer segmentation system using machine learning models.")
