import streamlit as st
import pandas as pd
import plotly.express as px

fabricated_data = pd.read_csv('data/fabricated_data.csv')
predicted_data = pd.read_csv('data/predicted_data.csv')

fabricated_data["DateTime"] = pd.to_datetime(fabricated_data["DateTime"])
predicted_data["DateTime"] = pd.to_datetime(predicted_data["DateTime"])

st.title('CPU Usage Visualization')

tab1, tab2 = st.tabs(["Fabricated CPU Usage", "Predicted CPU Usage"])

with tab1:
    fig1_a = px.line(fabricated_data, x='DateTime', y='CPU_Usage', 
                   markers=True)
    fig1_a.update_layout(xaxis_title='Time', yaxis_title='CPU Usage (%)')
    st.plotly_chart(fig1_a)

    fig1_b = px.line(fabricated_data, x='DateTime', y='Disk_Usage', 
                   markers=True)
    fig1_b.update_layout(xaxis_title='Time', yaxis_title='Disk Usage (%)')
    st.plotly_chart(fig1_b)

with tab2:
    fig2_a = px.line(predicted_data, x='DateTime', y='Predicted_CPU_Usage', 
                   markers=True)
    fig2_a.update_layout(xaxis_title='Time', yaxis_title='CPU Usage (%)')
    st.plotly_chart(fig2_a)

    fig2_b = px.line(predicted_data, x='DateTime', y='Predicted_Disk_Usage', 
                   markers=True)
    fig2_b.update_layout(xaxis_title='Time', yaxis_title='Disk Usage (%)')
    st.plotly_chart(fig2_b)
