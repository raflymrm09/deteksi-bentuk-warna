# deteksi-bentuk-warna
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files
import math
``
# Upload gambar
uploaded = files.upload()
filename = next(iter(uploaded))
img = cv2.imread(filename)
``
# Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fungsi hitung sudut
def angle(pt1, pt2, pt3):
    a = np.linalg.norm(pt2 - pt3)
    b = np.linalg.norm(pt1 - pt3)
    c = np.linalg.norm(pt1 - pt2)
    if a * c == 0:
        return 0
    cos_angle = (a**2 + c**2 - b**2) / (2 * a * c)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle) * 180 / np.pi

# Ekstraksi data
hasil = []

# Proses kontur
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        sisi = len(approx)
        shape = "Tidak dikenal"

        if sisi == 3:
            shape = "Segitiga"
        elif sisi == 4:
            rasio = float(w) / h
            pts = approx.reshape(4, 2)
            angles = []
            for i in range(4):
                pt1 = pts[i]
                pt2 = pts[(i + 1) % 4]
                pt3 = pts[(i + 2) % 4]
                ang = angle(pt1, pt2, pt3)
                angles.append(ang)
            if all(80 <= a <= 100 for a in angles):
                shape = "Kotak" if not (0.95 <= rasio <= 1.05) else "Persegi"
            else:
                shape = "Trapesium"
        elif sisi > 5:
            shape = "Lingkaran"
        elif 5 <= sisi <= 6:
            shape = "Trapesium"

        # Warna
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [approx], -1, 255, -1)
        mean_val = cv2.mean(img, mask=mask)
        R, G, B = int(mean_val[2]), int(mean_val[1]), int(mean_val[0])

        # Simpan data
        hasil.append({
            "Bentuk": shape,
            "Sisi": sisi,
            "Rasio": round(float(w)/h, 2),
            "R": R,
            "G": G,
            "B": B
        })

        # Gambar label
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
        cv2.putText(img, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Tampilkan gambar hasil
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.title('Deteksi Bentuk & Ekstraksi Warna + Bentuk')
plt.show()

# Tampilkan tabel hasil
df = pd.DataFrame(hasil)
df
