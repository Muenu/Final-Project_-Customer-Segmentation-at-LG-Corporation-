#!/usr/bin/env python
# coding: utf-8

# In[1]:


##1. Data Cleaning

import pandas as pd

# Loading the data
print("Loading data...")
df = pd.read_excel('Online Retail.xlsx')

#Show basic info
print("Original data shape:", df.shape)
print("Columns:", df.columns.tolist())

# Cleaning the data
print("Cleaning data...")

# Remove rows without Customer ID
clean_df = df[df['CustomerID'].notna()]

# Remove negative quantities (returns)
clean_df = clean_df[clean_df['Quantity'] > 0]

# Remove zero prices
clean_df = clean_df[clean_df['UnitPrice'] > 0]

# Create total amount column
clean_df['TotalAmount'] = clean_df['Quantity'] * clean_df['UnitPrice']

# Save cleaned data
clean_df.to_csv('cleaned_customer_data.csv', index=False)

print("Cleaned data shape:", clean_df.shape)
print("Data cleaning complete! Saved as 'cleaned_customer_data.csv'")


# In[2]:


## 2.RFM Analysis 

from datetime import datetime, timedelta

# Loading cleaned data
print("Calculating RFM values...")
df = pd.read_csv('cleaned_customer_data.csv')

# Convert date column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Finding most recent date in data
latest_date = df['InvoiceDate'].max()
print("Latest date in data:", latest_date)

# Calculate RFM for each customer
rfm_data = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency  
    'TotalAmount': 'sum'     # Monetary
}).reset_index()

# Rename columns
rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print("RFM calculation complete!")
print("Number of customers:", len(rfm_data))
print("\nSample of RFM data:")
print(rfm_data.head())

# Save RFM data
rfm_data.to_csv('rfm_values.csv', index=False)
print("RFM data saved as 'rfm_values.csv'")


# In[3]:


## 3. Customer Segmentation

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loading RFM data
print("Starting customer segmentation...")
rfm = pd.read_csv('rfm_values.csv')

# Use only RFM values for clustering
X = rfm[['Recency', 'Frequency', 'Monetary']]

# Scaling the data (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding the best number of clusters
print("Finding optimal number of clusters...")
wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot to find the "elbow"
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Finding the Right Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_plot.png')
plt.show()

# Based on the plot, choose 5 clusters
print("Creating 5 customer segments...")
kmeans = KMeans(n_clusters=5, random_state=42)
rfm['Segment'] = kmeans.fit_predict(X_scaled)

# Name the segments based on their characteristics
segment_names = {
    0: 'Lost Customers',
    1: 'Loyal Customers', 
    2: 'Champions',
    3: 'At-Risk Customers',
    4: 'New Customers'
}

rfm['Segment_Name'] = rfm['Segment'].map(segment_names)

# Save results
rfm.to_csv('customer_segments.csv', index=False)

print("Segmentation complete!")
print("\nSegment distribution:")
print(rfm['Segment_Name'].value_counts())


# In[4]:


##Simple Analysis Tool


import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CustomerSegmenter:
    def __init__(self):
        self.segment_names = {
            0: 'Lost Customers - Need reactivation',
            1: 'Loyal Customers - Reward loyalty', 
            2: 'Champions - Exclusive offers',
            3: 'At-Risk Customers - Win back',
            4: 'New Customers - Nurture'
        }
        
        self.marketing_ideas = {
            0: ["Send win-back email", "Offer special discount", "Ask for feedback"],
            1: ["Cross-sell products", "Offer volume discount", "Send loyalty reward"],
            2: ["Early access to new products", "Request testimonials", "Referral program"],
            3: ["Personalized 'we miss you' offer", "Limited-time discount", "Product recommendations"],
            4: ["Welcome series", "Educational content", "First purchase follow-up"]
        }
    
    def predict_segment(self, recency, frequency, monetary):
        # Simple rules-based approach (no machine learning needed)
        if frequency == 1 and recency <= 30:
            return 4  # New Customer
        elif frequency >= 8 and monetary >= 2000 and recency <= 30:
            return 2  # Champion
        elif frequency >= 4 and recency <= 90:
            return 1  # Loyal Customer  
        elif monetary >= 1500 and recency > 90:
            return 3  # At-Risk
        else:
            return 0  # Lost Customer
    
    def get_recommendations(self, recency, frequency, monetary):
        segment = self.predict_segment(recency, frequency, monetary)
        
        result = {
            'segment_number': segment,
            'segment_name': self.segment_names[segment],
            'marketing_ideas': self.marketing_ideas[segment],
            'input_values': {
                'recency': recency,
                'frequency': frequency, 
                'monetary': monetary
            }
        }
        
        return result

# Demo the tool
if __name__ == "__main__":
    print("LG Customer Segmentation Tool")
    print("=" * 40)
    
    segmenter = CustomerSegmenter()
    
    # Test examples
    test_customers = [
        (15, 12, 3400),   # Should be Champion
        (120, 5, 2100),   # Should be At-Risk
        (25, 1, 180),     # Should be New Customer
        (200, 2, 300)     # Should be Lost Customer
    ]
    
    for i, (recency, frequency, monetary) in enumerate(test_customers):
        print(f"\nCustomer {i+1}:")
        print(f"  Recency: {recency} days, Frequency: {frequency}, Monetary: ${monetary}")
        
        result = segmenter.get_recommendations(recency, frequency, monetary)
        print(f"  Segment: {result['segment_name']}")
        print(f"  Marketing Ideas: {', '.join(result['marketing_ideas'])}")


# In[5]:


## 5. Analysis Report

import matplotlib.pyplot as plt

# Loading the segmentation results
print("Creating analysis report...")
segments = pd.read_csv('customer_segments.csv')

# Basic statistics
print("=" * 50)
print("CUSTOMER SEGMENTATION REPORT")
print("=" * 50)

print(f"Total customers analyzed: {len(segments)}")
print("\nSegment Distribution:")
segment_counts = segments['Segment_Name'].value_counts()
print(segment_counts)

# Calculating average RFM by segment
print("\nAverage RFM Values by Segment:")
segment_stats = segments.groupby('Segment_Name').agg({
    'Recency': 'mean',
    'Frequency': 'mean', 
    'Monetary': 'mean'
}).round(2)

print(segment_stats)

# Creating simple visualization
plt.figure(figsize=(12, 5))

# Segment distribution pie chart
plt.subplot(1, 2, 1)
segment_counts.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Customer Segments Distribution')

# Monetary value by segment
plt.subplot(1, 2, 2)
segment_stats['Monetary'].plot.bar()
plt.title('Average Spending by Segment')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('segmentation_report.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nReport generated successfully!")
print("Check 'segmentation_report.png' for visualizations")


# In[ ]:




