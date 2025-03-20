import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime
import plotly.graph_objects as go
from openai import OpenAI
import os
#import speech_recognition as sr
from io import BytesIO
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, AudioProcessorBase, WebRtcMode
import numpy as np
import cv2
import mediapipe as mp
 
MODEL="gpt-4o-mini"
client = OpenAI(api_key="your_api_key")

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to check if person is lying down
def is_lying_down(landmarks):
    points_of_interest = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 
                          'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
    y_coords = [landmarks[mp_pose.PoseLandmark[point].value].y for point in points_of_interest]
    y_range = np.ptp(y_coords)  # Peak-to-peak (max - min)

    return y_range < 0.25  # Adjust threshold as needed

# Video processing class for Streamlit WebRTC
class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert image for Mediapipe processing
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        # Draw landmarks and check lying down status
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if is_lying_down(results.pose_landmarks.landmark):
                cv2.putText(img, "Lying Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(img, "Not Lying Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Ensure the frame is returned with pose detection overlayed
        return frame.from_ndarray(img, format="bgr24")

    
# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("medical_data.db")
    cursor = conn.cursor()
 
    # Create tables for medical data, notifications, medications, and emergency contacts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            blood_pressure TEXT NOT NULL,
            heart_rate INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            abnormal_data TEXT NOT NULL,
            abnormal_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            medication_name TEXT NOT NULL,
            dosage TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emergency_contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            relationship TEXT NOT NULL,
            phone TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
   
    conn.commit()
    return conn, cursor
 
# Validate blood pressure input format
def validate_bp(bp):
    parts = bp.split('/')
    if len(parts) != 2:
        return False
    systolic, diastolic = parts
    if not (systolic.isdigit() and diastolic.isdigit()):
        return False
    systolic = int(systolic)
    diastolic = int(diastolic)
    return 50 <= systolic <= 250 and 30 <= diastolic <= 150
 
# Function to validate input fields
def validate_inputs(name, age, bp, hr):
    if not name:
        st.error("Patient name is required.")
        return False
    if not age.isdigit() or not (1 <= int(age) <= 120):
        st.error("Please enter a valid age (1-120).")
        return False
    if not validate_bp(bp):
        st.error("Please enter blood pressure in format systolic/diastolic (e.g., 120/80).")
        return False
    if not hr.isdigit() or not (30 <= int(hr) <= 200):
        st.error("Please enter a valid heart rate (30-200 bpm).")
        return False
    return True
 
# Submit data to the database
def submit_data(name, age, bp, hr, conn, cursor):
    cursor.execute("""
        INSERT INTO medical_data (name, age, blood_pressure, heart_rate)
        VALUES (?, ?, ?, ?)
    """, (name, int(age), bp, int(hr)))
    conn.commit()
 
    # Check for abnormalities
    abnormality = detect_abnormal_data(name, bp, hr)
    if abnormality:
        cursor.execute("""
            INSERT INTO notifications (name, abnormal_data, abnormal_type)
            VALUES (?, ?, ?)
        """, (name, abnormality['data'], abnormality['type']))
        conn.commit()
        st.success(f"Notification created for abnormal {abnormality['type']}!")
 
    st.success("Data submitted successfully!")
 
# Detect abnormal data points
def detect_abnormal_data(name, bp, hr):
    systolic, diastolic = map(int, bp.split('/'))
    hr = int(hr)
 
    if hr < 60 or hr > 100:
        return {"data": f"Heart Rate: {hr} bpm", "type": "Heart Rate"}
 
    if systolic > 140 or diastolic > 90:
        return {"data": f"Blood Pressure: {systolic}/{diastolic} mmHg", "type": "Blood Pressure"}
 
    return None
 
# View submitted data as a dataframe
def view_data(cursor):
    cursor.execute("SELECT * FROM medical_data")
    records = cursor.fetchall()
    if records:
        df = pd.DataFrame(records, columns=["ID", "Name", "Age", "Blood Pressure", "Heart Rate", "Timestamp"])
        st.dataframe(df)
        return df
    else:
        st.warning("No data found.")
        return pd.DataFrame()
 
# Log medication to the database
def log_medication(name, med_name, dosage, conn, cursor):
    cursor.execute("""
        INSERT INTO medications (patient_name, medication_name, dosage)
        VALUES (?, ?, ?)
    """, (name, med_name, dosage))
    conn.commit()
 
# View medications as a dataframe
def view_medications(cursor):
    cursor.execute("SELECT * FROM medications")
    records = cursor.fetchall()
    if records:
        df = pd.DataFrame(records, columns=["ID", "Patient Name", "Medication Name", "Dosage", "Timestamp"])
        st.dataframe(df)
        return df
    else:
        st.warning("No medications logged.")
        return pd.DataFrame()
 
# Plot heart rate and blood pressure
def plot_data(df):
    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
       
        # Plot heart rate
        fig_hr = px.line(df, x='Timestamp', y='Heart Rate', title='Heart Rate Over Time', markers=True)
        st.plotly_chart(fig_hr)
 
        # Split blood pressure into systolic and diastolic
        df['Systolic'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
        df['Diastolic'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))
 
        # Plot blood pressure
        fig_bp = px.line(df, x='Timestamp', y=['Systolic', 'Diastolic'],
                         title='Blood Pressure Over Time', markers=True)
        st.plotly_chart(fig_bp)
 
# Notifications Page
def notifications_page(cursor):
    st.write("### Notifications")
 
    # Fetch notifications from the database
    cursor.execute("SELECT * FROM notifications ORDER BY timestamp DESC")
    notifications = cursor.fetchall()
 
    if notifications:
        for notif in notifications:
            st.write(f"**{notif[3]}** - **{notif[1]}** ({notif[2]})")
    else:
        st.info("No notifications available.")
 
    # Simulate a map with the user's current location and the nearest hospital
    st.write("### Nearby Hospital Route")
   
    # Example location (lat, lon) - Simulating user location
    user_location = [35.862213456659084, -78.8873404331946]  
    nearest_hospital = [35.7422, -78.7811]  # Wakemed Cary Hospital
   
    # Create map with user location and hospital
    map_ = folium.Map(location=user_location, zoom_start=14)
   
    # Add markers for user location and hospital
    folium.Marker(user_location, tooltip="Your Location", icon=folium.Icon(color="blue")).add_to(map_)
    folium.Marker(nearest_hospital, tooltip="Nearest Hospital", icon=folium.Icon(color="red")).add_to(map_)
   
    # Draw route (simulated)
    folium.PolyLine(locations=[user_location, nearest_hospital], color="green", weight=2.5).add_to(map_)
   
    # Display map in Streamlit
    st_folium(map_, width=700, height=500)
 
# Submit emergency contact to the database
def submit_contact(name, relationship, phone, conn, cursor):
    cursor.execute("""
        INSERT INTO emergency_contacts (name, relationship, phone)
        VALUES (?, ?, ?)
    """, (name, relationship, phone))
    conn.commit()
    st.success("Emergency contact added successfully!")
 
# View emergency contacts as a dataframe
def view_contacts(cursor):
    cursor.execute("SELECT * FROM emergency_contacts")
    records = cursor.fetchall()
    if records:
        df = pd.DataFrame(records, columns=["ID", "Name", "Relationship", "Phone", "Timestamp"])
        st.dataframe(df)
    else:
        st.warning("No emergency contacts found.")
 
# Function to handle chatbot responses using OpenAI GPT
def get_medical_response(user_input):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a healthcare bot. Help me with my health choices!"},  # System message providing context
            {"role": "user", "content": user_input}  # User input for the model
        ]
    )
    return completion.choices[0].message.content
 
 
# Function to display chatbot interface
def medical_chatbot():
    st.title("Medical Chatbot")
 
    st.write("Ask me anything related to medical or health concerns, and I'll try my best to assist you!")
   
    # User can input text or record audio directly
    user_input = st.text_input("Your Question")
 
    if user_input:
        with st.spinner("Thinking..."):
            response = get_medical_response(user_input)
        st.write(f"**Chatbot Response:** {response}")
 
# Main Streamlit app logic
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.selectbox("Go to", ["User Information", "Medical Data", "Medication Tracker", "Emergency Contacts", "AI Motion Detector", "Notifications", "Learning", "Medical Chatbot"])
 
    # Initialize the database
    conn, cursor = init_db()
 
    if selection == "User Information":  # Updated to "User Information"
        st.title("User Information Form")  # Updated title
 
        # Form inputs
        name = st.text_input("Name")
        age = st.text_input("Age")
        birthdate = st.date_input("Birthdate")  # Input for birthdate
        phone = st.text_input("Phone Number")
        email = st.text_input("Email")
        medication = st.text_input("Medication")
        allergies = st.text_input("Allergies")
        chronic_conditions = st.text_area("Chronic Conditions")  # Input for chronic conditions
        procedures = st.text_area("Medical Procedures")  # Input for medical procedures
        insurance = st.text_input("Insurance Information")
 
        # Submit button to handle user info submission
        if st.button("Submit"):
            if not name or not age or not phone or not email:
                st.error("Please fill in all required fields (Name, Age, Phone, and Email).")
            else:
                st.success("User information submitted successfully!")
 
        # Display the entered data back to the user
        if name and age and phone and email:
            st.write(f"**Name**: {name}")
            st.write(f"**Age**: {age}")
            st.write(f"**Birthdate**: {birthdate}")
            st.write(f"**Phone**: {phone}")
            st.write(f"**Email**: {email}")
            st.write(f"**Medication**: {medication}")
            st.write(f"**Allergies**: {allergies}")
            st.write(f"**Chronic Conditions**: {chronic_conditions}")
            st.write(f"**Medical Procedures**: {procedures}")
            st.write(f"**Insurance Info**: {insurance}")
 
    elif selection == "Medical Data":
        st.title("Medical Data")
 
        # Use columns for neater input layout
        col1, col2 = st.columns(2)
 
        with col1:
            name = st.text_input("Patient Name")
            age = st.text_input("Age")
 
        with col2:
            bp = st.text_input("Blood Pressure (mmHg)", placeholder="e.g., 120/80")
            hr = st.text_input("Heart Rate (bpm)")
 
        # Submit button
        if st.button("Submit"):
            if validate_inputs(name, age, bp, hr):
                submit_data(name, age, bp, hr, conn, cursor)
 
        # Display submitted data
        st.write("### Submitted Medical Data")
        df = view_data(cursor)
 
        # Plot heart rate and blood pressure with abnormal data highlighted
        if not df.empty:
            st.write("### Visualizations")
            plot_data(df)
 
    elif selection == "Medication Tracker":
        st.title("Medication Tracker")
 
        # Input form for medication logging
        name = st.text_input("Patient Name")
        med_name = st.text_input("Medication Name")
        dosage = st.text_input("Dosage")
 
        if st.button("Log Medication"):
            if name and med_name and dosage:
                log_medication(name, med_name, dosage, conn, cursor)
                st.success(f"{med_name} logged for {name} successfully!")
            else:
                st.error("Please fill in all fields.")
 
        # Display medication history
        st.write("### Medication History")
        df_med = view_medications(cursor)
 
    elif selection == "Emergency Contacts":
        st.title("Emergency Contacts")
 
        # Input form for emergency contact
        contact_name = st.text_input("Contact Name")
        relationship = st.text_input("Relationship")
        phone = st.text_input("Phone Number")
 
        if st.button("Add Emergency Contact"):
            if contact_name and relationship and phone:
                submit_contact(contact_name, relationship, phone, conn, cursor)
            else:
                st.error("Please fill in all fields.")
 
        # Display emergency contacts
        st.write("### Emergency Contacts List")
        view_contacts(cursor)
 
    elif selection == "AI Motion Detector":
        st.title("AI Motion Detector")
    
        # Start WebRTC stream and process video using PoseVideoProcessor
        webrtc_streamer(
            key="pose-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=PoseVideoProcessor,
            media_stream_constraints={"video": True, "audio": False}
        )

    elif selection == "Notifications":
        notifications_page(cursor)
 

    elif selection == "Learning":
        st.title("Learning About Heart Health")
        st.write("""
        ### Common Heart and Circulatory Problems
        1. **Angina**: Chest pain from reduced blood flow to the heart.
        2. **Shortness of Breath**: Difficulty breathing during physical activity.
        3. **Heart Attack**: Can occur due to coronary artery disease.
        4. **Arrhythmias**: Abnormal heart rhythms that can happen for various reasons.
        5. **Anemia**: Low red blood cells, possibly from poor nutrition, infections, or blood loss.
        6. **Atherosclerosis**: Hardening of arteries due to fatty deposits, leading to narrow or blocked blood vessels.
        7. **Congestive Heart Failure**: Common in older adults, particularly those over 75.
        8. **Coronary Artery Disease**: Often caused by atherosclerosis.
        9. **High Blood Pressure**: More common with age; medication should be managed with a doctor.
        10. **Heart Valve Diseases**: Aortic stenosis is a common condition.
        11. **Transient Ischemic Attacks (TIAs)**: Temporary disruptions in blood flow to the brain can lead to strokes.
        12. **Other Issues**:
            - Blood clots
            - Deep vein thrombosis
            - Peripheral vascular disease (pain in the legs while walking)
            - Varicose veins
            - Aneurysms (bulging arteries that can burst and cause serious issues)
        """)
 
        st.write("""
        ### Prevention Tips
        1. **Manage Risk Factors**: Control high blood pressure, cholesterol, diabetes, obesity, and avoid smoking.
        2. **Eat Healthy**: Follow a heart-healthy diet with low saturated fats and cholesterol. Keep a healthy weight.
        3. **Exercise Regularly**: Helps with weight control, diabetes management, and overall heart health. Start with moderate activity and consult your doctor first.
        4. **Regular Check-Ups**:
            - Get your blood pressure checked annually.
            - If you have diabetes or heart disease, monitor more frequently.
            - Check cholesterol every 5 years if it's normal; more often if you have certain conditions.
        """)
 
        st.write("By following these guidelines, you can support your heart and circulatory health!")
           
    elif selection == "Medical Chatbot":
        # Call the chatbot function here
        medical_chatbot()
 
if __name__ == "__main__":
    main()