import streamlit as st
import pandas as pd
import plotly.express as px

# Load the fabricated and predicted datasets
fabricated_data = pd.read_csv('data/fabricated_data.csv')
predicted_data = pd.read_csv('data/predicted_data.csv')

# Convert DateTime columns to datetime format
fabricated_data["DateTime"] = pd.to_datetime(fabricated_data["DateTime"])
predicted_data["DateTime"] = pd.to_datetime(predicted_data["DateTime"])

# Set the number of top nodes to display
top_n_nodes = 5  # Adjust this number as needed

# Get the top times with the highest Predicted Number of Nodes
top_nodes = predicted_data.nlargest(top_n_nodes, 'Predicted_Number_of_Nodes')[['DateTime', 'Predicted_Number_of_Nodes']]

# Extract only the time part
top_nodes['Time'] = top_nodes['DateTime'].dt.time.astype(str).str.slice(start=0, stop=5)

# Rearrange the columns
top_nodes = top_nodes[['Time', 'Predicted_Number_of_Nodes']]  # Select only the relevant columns

# Streamlit title and tabs
st.title('CPU, Disk Usage, and Number of Nodes Visualization')

tab1, tab2 = st.tabs(["Fabricated Data", "Predicted Data"])

# Fabricated data visualizations
with tab1:
    # CPU Usage plot
    fig1_a = px.line(fabricated_data, x='DateTime', y='CPU_Usage', markers=True)
    fig1_a.update_layout(xaxis_title='Time', yaxis_title='CPU Usage (%)')
    st.plotly_chart(fig1_a)

    # Disk Usage plot
    fig1_b = px.line(fabricated_data, x='DateTime', y='Disk_Usage', markers=True)
    fig1_b.update_layout(xaxis_title='Time', yaxis_title='Disk Usage (%)')
    st.plotly_chart(fig1_b)

    # Number of Nodes plot
    fig1_c = px.line(fabricated_data, x='DateTime', y='Number_of_Nodes', markers=True)
    fig1_c.update_layout(xaxis_title='Time', yaxis_title='Number of Nodes')
    st.plotly_chart(fig1_c)

# Predicted data visualizations
with tab2:
    # # Predicted CPU Usage plot
    # fig2_a = px.line(predicted_data, x='DateTime', y='Predicted_CPU_Usage', markers=True)
    # fig2_a.update_layout(xaxis_title='Time', yaxis_title='Predicted CPU Usage (%)')
    # st.plotly_chart(fig2_a)

    # # Predicted Disk Usage plot
    # fig2_b = px.line(predicted_data, x='DateTime', y='Predicted_Disk_Usage', markers=True)
    # fig2_b.update_layout(xaxis_title='Time', yaxis_title='Predicted Disk Usage (%)')
    # st.plotly_chart(fig2_b)

    # Predicted Number of Nodes plot
    fig2_c = px.line(predicted_data, x='DateTime', y='Predicted_Number_of_Nodes', markers=True)
    fig2_c.update_layout(xaxis_title='Time', yaxis_title='Predicted Number of Nodes')
    st.plotly_chart(fig2_c)

    # Display the top nodes with times and values
    st.subheader("Top Times with Highest Predicted Number of Nodes")
    st.table(top_nodes)
