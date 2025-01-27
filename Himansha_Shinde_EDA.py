import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')  # Set backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create visualizations directory
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Load the datasets
def load_data():
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert date columns to datetime
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    return customers_df, products_df, transactions_df

def perform_eda(customers_df, products_df, transactions_df):
    # Create a single figure with 6 subplots
    plt.figure(figsize=(20, 12))
    
    # 1. Customer Analysis
    print("\nCustomer Analysis:")
    print("Total customers:", len(customers_df))
    print("\nCustomers by Region:")
    print(customers_df['Region'].value_counts())
    
    # Plot 1: Customer Signup Trend
    plt.subplot(2, 3, 1)
    customers_df['SignupDate'].dt.year.value_counts().sort_index().plot(kind='bar')
    plt.title('Customer Signups by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Signups')
    
    # Plot 2: Customers by Region
    plt.subplot(2, 3, 2)
    customers_df['Region'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Customer Distribution by Region')
    
    # 2. Product Analysis
    print("\nProduct Analysis:")
    print("Total products:", len(products_df))
    print("\nProducts by Category:")
    print(products_df['Category'].value_counts())
    
    # Plot 3: Product Categories
    plt.subplot(2, 3, 3)
    products_df['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Product Category Distribution')
    
    # Plot 4: Product Price Distribution
    plt.subplot(2, 3, 4)
    sns.histplot(data=products_df, x='Price', bins=30)
    plt.title('Product Price Distribution')
    plt.xlabel('Price')
    
    # Plot 5: Transaction Value Distribution
    plt.subplot(2, 3, 5)
    sns.histplot(data=transactions_df, x='TotalValue', bins=30)
    plt.title('Transaction Value Distribution')
    plt.xlabel('Transaction Value')
    
    # Plot 6: Transaction Quantity Distribution
    plt.subplot(2, 3, 6)
    sns.histplot(data=transactions_df, x='Quantity', bins=30)
    plt.title('Transaction Quantity Distribution')
    plt.xlabel('Quantity')
    
    plt.tight_layout()
    plt.savefig('visualizations/eda_complete.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_customer_features(customers_df, transactions_df, products_df):
    # Calculate customer metrics
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'TotalValue': ['sum', 'mean'],
        'Quantity': ['sum', 'mean']
    })
    
    # Flatten column names
    customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns.values]
    
    # Calculate category preferences
    customer_categories = transactions_df.merge(products_df, on='ProductID')
    category_pivot = pd.crosstab(customer_categories['CustomerID'], 
                                customer_categories['Category'])
    
    # Combine features
    customer_features = pd.merge(customer_metrics, category_pivot, 
                               left_index=True, right_index=True, how='left')
    
    # Add customer region (one-hot encoded)
    region_dummies = pd.get_dummies(customers_df.set_index('CustomerID')['Region'])
    customer_features = pd.merge(customer_features, region_dummies,
                               left_index=True, right_index=True, how='left')
    
    # Fill any NaN values with 0
    customer_features = customer_features.fillna(0)
    
    return customer_features

def find_lookalikes(customer_features, target_customer_id, n_recommendations=3):
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(customer_features)
    
    # Calculate similarity
    similarity_matrix = cosine_similarity(features_scaled)
    similarity_df = pd.DataFrame(similarity_matrix, 
                               index=customer_features.index,
                               columns=customer_features.index)
    
    # Get top similar customers
    similar_customers = similarity_df[target_customer_id].sort_values(ascending=False)[1:n_recommendations+1]
    
    return similar_customers

def perform_clustering(customer_features, max_clusters=10):
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(customer_features)
    
    # Find optimal number of clusters using elbow method
    inertias = []
    db_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)
        db_scores.append(davies_bouldin_score(features_scaled, kmeans.labels_))
    
    # Create a single figure with clustering analysis
    plt.figure(figsize=(20, 10))
    
    # Plot 1: Elbow Method
    plt.subplot(1, 3, 1)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    # Plot 2: Davies-Bouldin Score
    plt.subplot(1, 3, 2)
    plt.plot(range(2, max_clusters + 1), db_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score vs. Number of Clusters')
    
    # Select optimal number of clusters based on DB score
    optimal_clusters = db_scores.index(min(db_scores)) + 2
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = final_kmeans.fit_predict(features_scaled)
    
    # Plot 3: PCA Visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=cluster_labels, cmap='viridis')
    plt.title('Customer Segments (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster')
    
    plt.tight_layout()
    plt.savefig('visualizations/clustering_complete.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_labels, min(db_scores), optimal_clusters

def main():
    # Load data
    print("Loading data...")
    customers_df, products_df, transactions_df = load_data()
    
    # Task 1: EDA
    print("\nPerforming EDA...")
    perform_eda(customers_df, products_df, transactions_df)
    
    # Task 2: Lookalike Model
    print("\nCreating customer features...")
    customer_features = create_customer_features(customers_df, transactions_df, products_df)
    
    print("\nGenerating lookalikes for customers C0001-C0020...")
    lookalike_results = []
    for cust_id in customers_df['CustomerID'][:20]:
        similar_customers = find_lookalikes(customer_features, cust_id)
        for rank, (similar_id, score) in enumerate(similar_customers.items(), 1):
            lookalike_results.append({
                'customer_id': cust_id,
                'rank': rank,
                'similar_customer': similar_id,
                'similarity_score': score
            })
    
    # Save lookalike results
    lookalike_df = pd.DataFrame(lookalike_results)
    lookalike_df.to_csv('Himansha_Lookalike.csv', index=False)
    
    # Task 3: Customer Segmentation
    print("\nPerforming customer segmentation...")
    cluster_labels, db_score, n_clusters = perform_clustering(customer_features)
    
    # Save clustering results
    clustering_results = pd.DataFrame({
        'CustomerID': customer_features.index,
        'Cluster': cluster_labels
    })
    clustering_results.to_csv('Himansha_Clustering_Results.csv', index=False)
    
    # Save clustering summary
    clustering_summary = pd.DataFrame({
        'Metric': ['Number of Clusters', 'Davies-Bouldin Score', 'Total Customers'],
        'Value': [n_clusters, db_score, len(cluster_labels)]
    })
    clustering_summary.to_csv('Himansha_Clustering_Summary.csv', index=False)
    
    print("\nAnalysis complete! Check the generated files and visualizations directory for results.")

if __name__ == "__main__":
    main()