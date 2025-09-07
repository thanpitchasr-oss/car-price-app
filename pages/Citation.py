import streamlit as st

st.title("📝 Citation")

# ใช้ st.markdown กับ Markdown แยกบรรทัด
st.markdown("""
##การอ้างอิง##  

นางสาวธัญพิชชา ศรีเพชร รหัสนักศึกษา 6530206035  
นางสาววรฤทัย ตานำคำ รหัสนักศึกษา 6530206124  
อาจารย์ที่ปรึกษาหลัก: ผศ.ดร.อานิษา ราศรี  

**หัวข้อโครงงาน:**  
การทำนายราคารถยนต์มือสองด้วยขั้นตอนวิธี CatBoost และ LightGBM พร้อมต่อยอดสู่การใช้งานเว็บแอปพลิเคชันเพื่อแสดงผล  

**English title:**  
Used Car Price Prediction Using CatBoost and LightGBM Algorithms with Web Application Deployment for Result Visualization
""")

# แสดงภาพ
st.image("https://i.pinimg.com/1200x/d1/10/ca/d110caddb4e65605339c06e792ca59f9.jpg")
