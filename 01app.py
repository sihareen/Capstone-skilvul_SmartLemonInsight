import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Definisikan arsitektur model (harus sesuai dengan model yang disimpan)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(3 * 224 * 224, 50)  # Sesuaikan dengan arsitektur model Anda
        self.layer2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Memuat model
model = MyModel()
model.load_state_dict(torch.load('best60.pt'))
model.eval()

# Fungsi untuk memproses gambar
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Tambahkan batch dimension
    return image

# Fungsi untuk melakukan prediksi
def predict(image):
    image_tensor = process_image(image)
    with torch.no_grad():
        output = model(image_tensor)
    return output.numpy()

# Antarmuka Streamlit
st.title('Image Classification App')
st.write('Upload an image to classify:')

# Upload gambar dari pengguna
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = predict(image)
    st.write(f'Prediction: {prediction}')
