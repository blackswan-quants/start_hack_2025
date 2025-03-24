import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Sample dataframe, replace with your actual df
# Assuming df has columns like 'x', 'y', 'feature1', 'feature2', etc.

# Read your data
df = pd.read_pickle("averaged.pkl")
df = pd.DataFrame(df)

# Streamlit layout
st.title("Coordinate and Feature Visualization Dashboard")

# Sidebar for user inputs
st.sidebar.header("Filters")

# Dropdown for selecting which feature to visualize
selected_feature = st.sidebar.selectbox("Select feature to display", df.columns[2:])

# Display the dataframe
st.subheader("Data Preview")
st.dataframe(df)

# Plot the coordinates and selected feature
st.subheader(f"Scatter Plot of Coordinates with {selected_feature}")
fig = px.scatter(df, x='x', y='y', color=selected_feature, title=f"Scatter Plot: {selected_feature} vs Coordinates")
st.plotly_chart(fig)

# Optional: Show a heatmap of correlations between features
st.subheader("Feature Correlation Heatmap")
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Optional: Pairplot of all features (good for multivariate analysis)
st.subheader("Pairplot of Features")
sns.pairplot(df)
st.pyplot()

# Show a line plot for selected feature
st.subheader(f"Line Plot of {selected_feature}")
fig, ax = plt.subplots()
ax.plot(df['x'], df[selected_feature], marker='o', linestyle='-', color='b')
ax.set_title(f"{selected_feature} vs x")
ax.set_xlabel("x")
ax.set_ylabel(selected_feature)
st.pyplot(fig)
