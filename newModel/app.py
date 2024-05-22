import tensorflow as tf
import cv2
import numpy as np

# Memuat model yang telah dilatih
model = tf.keras.models.load_model("best_model_at_epoch_21.keras")

# Fungsi untuk melakukan prediksi dan menggambar bounding box
def predict_and_draw_boxes(frame, model):
    # Preprocessing frame
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    
    # Melakukan prediksi
    predictions = model.predict(input_tensor)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1)
    
    # Menggambar bounding box dan label
    for i in range(len(predicted_class)):
        if confidence[i] > 0.5:  # Threshold untuk confidence
            label = f"Class: {predicted_class[i]}, Conf: {confidence[i]:.2f}"
            cv2.putText(frame, label, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Menginisialisasi webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Melakukan prediksi dan menggambar bounding box
    frame = predict_and_draw_boxes(frame, model)
    
    # Menampilkan frame
    cv2.imshow('Object Detection', frame)
    
    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan dan menutup jendela
cap.release()
cv2.destroyAllWindows()