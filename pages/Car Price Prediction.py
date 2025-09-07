import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load("catboost_model.joblib")

st.title("🚗 Car Price Prediction (USD)")
st.write("กรอกข้อมูลรถของคุณเพื่อทำนายราคารถยนต์มือสอง (USD)")

# Input
fuel_type = st.selectbox("ประเภทเชื้อเพลิง", ["Petrol (เบนซิน)", "Diesel (ดีเซล)", "Electric (ไฟฟ้า)"])
mileage = st.number_input("อัตราการสิ้นเปลืองน้ำมัน (5-35 กม./ลิตร)", min_value=5.0, value=5.0, step=0.01, max_value=35.0)
engine_cc = st.number_input("ขนาดเครื่องยนต์ (800-5000 cc)", min_value=800, value=800, step=100,max_value=5000)
car_age = st.number_input("อายุรถ (0-30 ปี)", min_value=0, value=0, step=1, max_value=30)
owner_count = st.selectbox("จำนวนเจ้าของก่อนหน้า", ["1", "2", "3", "4", "5"])
accidents_reported = st.selectbox("จำนวนอุบัติเหตุที่รายงาน (ครั้ง)", ["0","1","2","3","4","5"])
insurance_valid = st.selectbox("ประกันภัยยังคงใช้งานได้หรือไม่", ["ใช้งานได้", "หมดอายุ"])
color = st.selectbox("สีของรถยนต์", ["Red", "Blue", "Silver", "Black", "White", "Gray"])
tranmission = st.selectbox("ประเภทเกียร์", ["อัตโนมัติ", "ธรรมดา"])
brand = st.selectbox("ยี่ห้อรถยนต์", ["Nissan", "Volkswagen", "BMW", "Tesla", "Honda", "Chevrolet", "Hyundai", "Toyota", "Kia", "Ford"])
service_history = st.selectbox("ประวัติการบำรุงรักษา", ["Full (ครบถ้วน)", "Partial (บางส่วน)", "None (ไม่มี)"])

# ✅ Mapping (encode ให้ตรงกับที่ train)
fuel_map = {"Petrol (เบนซิน)": 2, "Diesel (ดีเซล)": 1, "Electric (ไฟฟ้า)": 0}
insurance_map = {"ใช้งานได้": 1, "หมดอายุ": 0}
service_map = {"None (ไม่มี)": 0, "Partial (บางส่วน)": 1, "Full (ครบถ้วน)": 2}
trans_map = {"อัตโนมัติ": 1, "ธรรมดา": 0}
color_map = {"Red": 1, "Blue": 0, "Silver": 2, "Black": 3, "White": 5, "Gray": 4}
brand_map = {"Nissan":6,"Volkswagen":2,"BMW":1,"Tesla":0,"Honda":5,"Chevrolet":8,"Hyundai":4,"Toyota":7,"Kia":3,"Ford":9}

fuel_type_encoded = fuel_map[fuel_type]
insurance_encoded = insurance_map[insurance_valid]
service_encoded = service_map[service_history]
trans_encoded = trans_map[tranmission]
color_encoded = color_map[color]
brand_encoded = brand_map[brand]
owner_count = int(owner_count)
accidents_reported = int(accidents_reported)

# ✅ รวมเป็น array (เรียงตาม features ตอน train)
input_data = np.array([[mileage, engine_cc, car_age, owner_count,
                        accidents_reported, insurance_encoded, 
                        service_encoded, fuel_type_encoded,
                        brand_encoded, trans_encoded, color_encoded]])

import time
if st.button("ทำนายราคา"):
    with st.spinner('กำลังคิดราคารถของคุณ...'):
        time.sleep(4)  # เพิ่ม delay ให้ดูเหมือนกำลังประมวลผล
        prediction = model.predict(input_data)
    st.success(f"✅ ราคาที่ทำนายได้: {prediction[0]:,.2f} USD")
