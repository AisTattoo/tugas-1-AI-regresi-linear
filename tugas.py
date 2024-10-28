import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Membuat data contoh
# X adalah fitur (variabel independen)
# y adalah target (variabel dependen)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 5, 7, 11])

# Membuat model regresi linear
model = LinearRegression()
model.fit(X, y)

# Memprediksi nilai dengan model
y_pred = model.predict(X)

# Menampilkan koefisien dan intercept
print(f'Koefisien: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')

# Visualisasi
plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, y_pred, color='red', label='Regresi Linear')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresi Linear Sederhana')
plt.legend()
plt.show()
