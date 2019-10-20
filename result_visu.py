import matplotlib.pyplot as plt

"""
x-axis : epoch
y-axis : LAS - accuracy WITHOUT CONFIDENCE (in blue)
         LAS - accuracy WITH CONFIDENCE (in red)
         
Last graphs : With/without levenshtein embeddings
"""

# en-fr 100-100

Y_bas = [0.92, 2.19, 1.93, 1.92, 2.71, 5.74, 6.02, 8.95, 12.55, 18.43, 18.81, 22.75, 26.43, 27.66, 32.95, 32.83, 34.92, 36.43, 38.74, 41.52, 43.4, 41.73, 45.06, 46.2, 46.45, 45.75, 46.7, 49.06, 49.64, 49.89, 49.32, 49.18, 50.42, 51.25, 51.52, 50.61, 51.45, 52.53, 52.79, 52.22, 52.2, 52.18, 52.66, 51.55, 50.65, 50.94, 52.32, 52.39, 52.97, 53.66, 54.17, 53.97, 53.89, 53.56, 54.71, 53.06, 53.08, 53.05, 53.58, 53.26, 53.18, 53.92, 54.02, 54.36, 54.04, 55.08, 55.34, 53.2, 53.06, 54.56, 53.96, 52.09, 53.24, 54.07, 54.09, 54.22, 53.95, 54.07, 53.63, 53.49, 52.27, 52.98, 52.9, 53.82, 53.9, 54.62, 54.91, 55.0, 55.2, 55.24, 55.36, 55.19, 54.31, 54.61, 55.44, 54.27, 53.98, 54.6, 54.28, 53.41, 54.07, 53.98, 54.68, 55.57, 55.85, 54.58, 54.34, 53.8, 53.47, 54.16, 52.95, 53.03, 53.91, 54.72, 54.01, 54.33, 54.89, 54.33, 54.27, 54.3, 54.34, 54.13, 54.78, 54.4, 54.55, 54.73, 55.07, 55.34, 55.14, 54.24, 54.26, 55.22, 55.39, 55.03, 55.6, 55.55, 56.26, 55.69, 55.07, 55.96, 56.33, 55.84, 55.67, 56.21, 55.4, 54.35, 54.19, 53.98, 54.28, 54.23, 54.44, 55.48, 56.23, 56.51, 56.49, 56.65, 56.0, 56.34, 55.91, 55.86, 55.67, 56.71, 56.68, 57.09, 56.8, 56.86, 56.35, 55.86, 55.18, 55.64, 56.04, 54.65, 56.43, 56.46, 55.95, 55.69, 56.02, 55.59, 54.25, 54.03, 54.83, 54.98, 54.38, 54.84, 54.35, 54.05, 54.6, 55.38, 55.67, 56.54, 56.66, 55.93, 55.88, 55.7, 55.35, 55.22, 55.99, 56.82, 56.88, 57.24, 56.98, 56.76, 56.2, 56.42, 55.53, 54.78, 52.68, 52.28, 53.72, 54.31, 54.69, 52.83, 53.07, 54.1, 54.43, 53.48, 54.28, 54.74, 54.84, 54.83, 55.43, 55.59, 55.09, 55.2, 55.4, 55.94, 55.41, 54.65, 55.25, 55.79, 55.86, 55.74, 55.35, 55.68, 55.8, 56.06, 55.96, 55.82, 55.83, 55.87, 55.51, 55.39, 54.53, 54.65, 55.27, 55.44, 56.15, 53.62, 54.35]

Y_shift = [0.69, 1.07, 1.81, 1.14, 3.0, 5.3, 5.77, 9.52, 11.09, 15.94, 21.19, 23.74, 24.82, 25.24, 28.08, 32.04, 29.21, 33.71, 33.55, 34.98, 36.84, 37.64, 40.88, 41.72, 40.58, 41.62, 45.34, 44.94, 45.24, 47.28, 47.06, 48.31, 49.22, 49.59, 47.55, 49.13, 48.66, 48.58, 48.84, 49.86, 51.0, 50.05, 50.71, 50.92, 51.2, 51.6, 51.65, 51.79, 52.01, 51.55, 51.97, 52.91, 51.83, 52.12, 52.02, 51.89, 51.11, 52.38, 50.96, 51.23, 51.35, 50.63, 48.23, 49.14, 51.21, 47.33, 49.63, 51.0, 51.05, 51.86, 51.45, 51.29, 51.8, 52.49, 51.53, 51.95, 52.32, 51.52, 51.14, 52.02, 52.07, 52.41, 53.59, 53.18, 53.16, 53.28, 53.77, 54.26, 55.21, 54.92, 53.99, 53.58, 53.66, 53.55, 53.5, 52.5, 52.8, 53.32, 53.38, 53.33, 53.95, 54.43, 54.77, 54.74, 55.2, 55.8, 55.28, 54.77, 54.69, 54.47, 54.85, 54.06, 53.43, 52.67, 52.65, 51.22, 53.42, 53.53, 53.04, 52.91, 53.25, 52.53, 53.2, 52.95, 53.68, 53.0, 52.76, 53.25, 53.78, 53.48, 53.16, 53.35, 53.18, 54.23, 53.85, 53.69, 54.13, 53.88, 54.26, 54.29, 54.77, 54.3, 54.75, 54.62, 55.16, 55.07, 54.99, 54.81, 54.67, 55.03, 55.01, 54.23, 53.21, 53.56, 54.67, 53.75, 53.78, 53.85, 54.4, 54.5, 54.34, 54.23, 54.79, 54.59, 55.34, 55.22, 54.76, 55.11, 55.6, 55.96, 55.84, 55.97, 55.71, 55.57, 55.27, 54.98, 55.23, 55.32, 54.81, 55.25, 54.76, 55.47, 55.63, 55.69, 54.72, 54.24, 54.32, 53.53, 53.76, 53.87, 53.63, 53.59, 52.49, 54.5, 54.51, 54.08, 54.59, 54.11, 54.65, 54.56, 55.5, 55.02, 55.14, 55.51, 55.99, 56.15, 55.36, 55.13, 54.98, 55.29, 55.32, 55.88, 55.67, 55.81, 55.99, 55.66, 55.27, 54.9, 54.23, 53.6, 54.28, 53.74, 54.13, 53.9, 53.24, 52.87, 53.47, 54.08, 54.85, 54.67, 54.82, 55.33, 55.91, 55.82, 55.96, 56.46, 56.66, 56.62, 56.5, 56.39, 56.29, 56.24, 55.65, 55.71, 55.75, 55.42, 53.66, 54.34, 54.7]


X = list(range(len(Y_bas)))

plt.plot(X, Y_bas, color='blue')
plt.plot(X, Y_shift, color='red')
plt.title("en-fr 100-100")
plt.show()

# en-fr 1000-100

Y_bas = [11.8, 20.81, 31.99, 38.4, 41.5, 46.19, 47.58, 50.46, 52.09, 53.58, 54.66, 57.35, 55.25, 55.28, 56.46, 56.32, 56.7, 57.52, 57.17, 58.0, 57.68, 58.61, 58.04, 58.08, 57.74, 57.91, 59.16, 57.47, 58.16, 58.86, 58.25, 57.69, 57.59, 58.7, 58.29, 58.84, 59.79, 58.89, 58.55, 58.67, 57.34, 58.49, 57.48, 59.24, 59.66, 58.99, 58.32, 57.42, 59.48, 59.03, 59.97, 59.46, 57.51, 58.47, 59.12, 59.95, 59.47, 60.13, 60.22, 60.36, 59.87, 58.12, 59.36, 58.81, 59.08, 58.33, 59.06, 59.66, 58.52, 58.88, 59.38, 59.81, 59.89, 59.56, 60.04, 59.12, 60.6, 59.43, 58.95, 59.66, 59.56, 59.07, 59.58, 58.87, 59.15, 59.47, 59.83, 59.25, 60.66, 59.89, 60.0, 59.58, 59.09, 59.19, 60.22, 59.6, 60.03, 59.63, 60.56, 60.97, 60.68, 61.25, 60.66, 60.86, 60.84, 60.64, 60.77, 59.82, 60.64, 61.34, 60.56, 61.3, 60.87, 60.11, 60.77, 60.36, 60.57, 61.95, 61.5, 60.47, 60.69, 61.01, 61.37, 60.66, 60.4, 60.42, 61.31, 59.8, 59.19, 59.56, 60.87, 60.83, 59.87, 60.36, 59.42, 60.07, 59.56, 60.08, 60.36, 60.55, 60.95, 59.98, 60.2, 60.6, 60.61, 59.92, 59.77, 60.41, 60.02, 60.41, 60.53, 59.63, 60.9, 60.26, 61.43, 61.0, 60.66, 60.94, 61.12, 60.39, 61.24, 60.91, 60.94, 60.91, 61.18, 60.7, 60.67, 60.47, 60.28, 60.34, 60.67, 60.53, 60.35, 60.4, 61.3, 60.8, 61.26, 61.49, 60.9, 60.53, 60.8, 61.12, 61.34, 61.05, 60.94, 60.62, 60.46, 60.68, 60.33, 60.87, 60.71, 60.59, 60.28, 60.56, 60.79, 60.01, 60.37, 60.64, 61.2, 60.84, 60.91, 61.58, 61.42, 61.5, 61.76, 62.4, 62.33, 61.81, 61.52, 61.23, 62.07, 61.59, 61.55, 61.86, 61.81, 61.26, 61.67, 61.46, 60.87, 61.73, 60.76, 60.78, 61.05, 61.1, 61.32, 61.61, 61.14, 61.53, 60.94, 60.97, 61.27, 61.55, 60.75, 60.88, 60.51, 60.79, 60.69, 60.09, 60.6, 61.43, 61.56, 61.97, 61.69, 61.17, 61.09, 61.27, 61.86, 62.06, 62.11]
Y_shift = [9.47, 25.99, 31.15, 37.99, 40.45, 44.72, 47.6, 50.19, 51.81, 52.15, 53.64, 53.23, 53.22, 53.84, 53.78, 54.11, 56.8, 54.41, 54.06, 56.15, 57.65, 56.7, 56.88, 58.11, 58.76, 57.71, 57.09, 57.29, 57.52, 57.64, 57.76, 58.4, 58.11, 57.69, 58.33, 59.49, 58.65, 57.57, 58.28, 58.97, 57.97, 57.37, 58.63, 58.99, 59.04, 60.42, 60.07, 59.6, 60.26, 59.64, 59.37, 60.04, 59.68, 59.03, 60.14, 60.32, 59.82, 60.68, 60.38, 60.86, 59.19, 59.63, 60.16, 59.51, 59.9, 60.35, 61.35, 60.33, 60.89, 60.02, 60.54, 60.5, 61.19, 60.82, 60.39, 60.96, 60.59, 61.06, 60.77, 60.61, 60.09, 60.3, 61.04, 60.14, 60.78, 58.68, 58.83, 59.62, 59.0, 59.8, 59.58, 60.36, 60.78, 60.13, 59.44, 60.59, 60.24, 60.88, 60.76, 60.51, 60.47, 60.44, 59.87, 59.81, 59.87, 60.98, 60.71, 60.68, 60.8, 61.51, 61.7, 60.19, 61.39, 61.06, 60.87, 61.61, 61.27, 61.99, 61.14, 61.64, 60.66, 60.57, 61.04, 61.35, 60.39, 60.64, 61.54, 60.58, 61.23, 61.27, 61.44, 61.58, 61.56, 61.95, 62.33, 61.32, 61.52, 60.63, 61.4, 61.35, 60.49, 61.3, 61.01, 60.68, 61.09, 61.27, 61.89, 61.79, 61.79, 61.87, 62.26, 61.87, 61.92, 62.05, 61.3, 62.12, 61.57, 61.34, 61.29, 61.33, 61.18, 61.65, 62.04, 60.97, 61.46, 62.37, 61.65, 62.39, 62.52, 62.41, 61.6, 61.72, 61.63, 61.6, 60.92, 61.75, 62.12, 61.17, 61.16, 61.55, 61.34, 61.94, 61.42, 61.47, 61.87, 62.25, 62.31, 62.45, 62.01, 61.82, 60.73, 61.24, 61.38, 61.05, 61.31, 61.16, 61.61, 61.41, 61.53, 61.93, 60.8, 61.11, 61.24, 61.25, 62.06, 61.8, 62.34, 61.76, 61.47, 62.03, 62.09, 62.33, 61.33, 61.25, 60.61, 61.5, 62.01, 61.17, 61.69, 62.08, 62.23, 61.46, 62.61, 62.19, 60.99, 60.97, 62.66, 62.37, 61.93, 62.02, 62.47, 62.37, 62.1, 63.02, 63.0, 62.39, 63.0, 62.99, 62.6, 62.45, 61.36, 61.75, 61.68, 62.0, 62.1, 62.91, 61.78, 62.11, 61.81]


X = list(range(len(Y_bas)))

plt.plot(X, Y_bas, color='blue')
plt.plot(X, Y_shift, color='red')
plt.title("en-fr 1000-100")
plt.show()


# en-fr 10000-100

Y_bas = [39.15, 49.69, 52.03, 55.67, 56.07, 55.99, 58.54, 57.1, 59.4, 59.47, 61.53, 59.94, 59.63, 59.17, 60.54, 59.44, 60.19, 60.52, 60.9, 61.85, 60.14, 61.04, 62.01, 62.16, 61.32, 61.75, 61.58, 61.86, 61.3, 62.1, 61.62, 61.61, 61.79, 62.75, 62.31, 61.71, 62.62, 62.51, 63.19, 62.78, 62.27, 62.71, 63.32, 62.9, 62.59, 62.41, 63.42, 62.2, 62.22, 63.13, 62.24, 63.3, 62.69, 62.33, 62.06, 63.12, 63.48, 64.09, 62.95, 63.13, 63.43, 63.09, 62.84, 62.94, 63.07, 63.63, 63.4, 63.41, 63.93, 63.65, 62.53, 63.97, 63.71, 63.58, 64.09, 63.63, 63.72, 63.88, 62.98, 63.95, 63.58, 64.05, 64.1, 63.67, 64.11, 63.62, 62.87, 63.47, 63.8, 64.09, 64.08, 64.14, 63.53, 63.18, 63.89, 63.62, 63.77, 64.07, 63.56, 63.59, 63.34, 63.39, 63.84, 64.18, 64.05, 63.92, 63.93, 63.37, 63.78, 63.2, 63.79, 64.03, 63.61, 63.87, 63.86, 63.99, 63.49, 63.52, 63.91, 63.86, 64.06, 64.55, 64.14, 64.03, 63.67, 64.51, 64.41, 64.46, 63.85, 63.23, 64.26, 64.06, 64.84, 64.37, 63.88, 63.66, 63.45, 63.89, 63.81, 63.81, 63.8, 63.87, 64.08, 63.69, 64.14, 63.97, 63.78, 64.05, 64.24, 64.29, 63.73, 63.95, 63.85, 64.45, 64.26, 64.15, 63.67, 63.73, 64.17, 63.95, 64.39, 63.95, 64.61, 64.13, 63.97, 63.78, 64.15, 64.31, 64.34, 64.09, 64.21, 64.03, 64.32, 64.47, 64.46, 64.41, 64.07, 64.42, 64.3, 64.52, 64.31, 64.64, 64.9, 64.1, 64.52, 64.65, 64.68, 64.61, 64.98, 64.72, 64.89, 64.77, 64.8, 64.82, 64.69, 65.05, 64.91, 64.73, 64.78, 65.09, 65.22, 64.9, 65.06, 64.83, 65.02, 64.81, 64.4, 64.76, 64.35, 65.13, 64.56, 64.3, 64.43, 64.37, 63.91, 64.72, 64.96, 64.83, 64.52, 64.48, 64.41, 64.46, 64.71, 65.05, 64.77, 64.78, 64.98, 64.75, 64.85, 64.95, 64.99, 64.91, 64.77, 64.99, 64.91, 64.83, 64.86, 64.74, 64.75, 64.76, 64.68, 64.83, 64.8, 64.97, 64.93, 64.74, 64.64, 64.97, 64.64]

Y_shift = [42.19, 47.19, 51.09, 53.03, 55.62, 56.34, 57.75, 56.91, 59.54, 56.97, 59.14, 60.0, 59.77, 59.17, 60.5, 60.25, 61.06, 60.66, 60.28, 61.58, 60.67, 61.09, 61.53, 60.95, 60.36, 60.61, 61.03, 60.08, 61.16, 61.27, 60.9, 61.71, 61.44, 61.93, 61.89, 62.02, 62.03, 62.38, 62.31, 62.3, 62.11, 62.27, 62.14, 62.69, 62.18, 61.64, 62.25, 61.85, 62.21, 62.08, 62.08, 62.76, 62.8, 62.35, 61.85, 62.68, 62.07, 61.61, 61.83, 62.97, 63.01, 63.11, 62.96, 62.68, 62.35, 62.57, 62.3, 62.63, 62.83, 62.32, 61.89, 62.06, 62.48, 62.02, 62.21, 62.68, 63.24, 63.1, 62.26, 62.76, 63.05, 62.5, 62.87, 63.04, 63.49, 63.95, 63.19, 63.82, 63.59, 63.07, 63.05, 63.7, 63.25, 63.59, 64.31, 63.73, 63.66, 62.95, 63.03, 62.77, 63.12, 63.19, 63.23, 63.39, 62.71, 62.61, 63.4, 62.71, 63.26, 63.08, 63.5, 63.22, 63.56, 62.98, 62.78, 63.34, 63.38, 63.96, 63.58, 63.45, 63.95, 63.62, 63.88, 63.29, 63.06, 63.88, 63.45, 63.47, 63.44, 63.47, 63.6, 63.69, 63.67, 63.43, 63.9, 64.05, 63.8, 64.05, 63.95, 64.07, 63.45, 63.95, 63.6, 63.61, 63.63, 64.07, 64.36, 64.27, 64.54, 64.85, 64.27, 64.08, 64.22, 63.91, 64.32, 63.78, 64.05, 64.23, 64.04, 63.81, 63.81, 63.66, 63.88, 63.89, 64.05, 63.84, 64.04, 64.32, 64.47, 64.39, 64.28, 64.57, 64.58, 64.6, 64.45, 64.08, 64.37, 64.17, 63.89, 63.95, 64.1, 64.29, 64.04, 63.94, 64.05, 63.84, 63.95, 63.81, 63.72, 63.87, 64.01, 64.0, 64.14, 64.26, 64.34, 64.07, 63.95, 64.31, 64.54, 64.48, 64.58, 64.46, 64.43, 64.48, 64.05, 64.09, 64.57, 64.38, 64.61, 64.53, 64.36, 64.56, 64.77, 64.79, 64.82, 64.92, 64.72, 64.74, 64.34, 64.45, 64.46, 64.53, 64.46, 64.79, 64.8, 64.85, 64.88, 64.55, 64.5, 64.44, 64.48, 64.65, 64.66, 64.73, 64.77, 64.68, 64.46, 64.54, 64.35, 64.43, 64.37, 64.47, 64.55, 64.62, 64.54, 64.34, 64.39, 64.55, 64.6]

X = list(range(len(Y_bas)))

plt.plot(X, Y_bas, color='blue')
plt.plot(X, Y_shift, color='red')
plt.title("en-fr 10000-100")
plt.show()


# en-fr 1000-1000

Y_bas = [31.54, 47.34, 58.97, 63.5, 68.03, 69.65, 71.07, 72.94, 73.63, 73.82, 74.89, 74.84, 75.13, 75.72, 75.61, 76.29, 76.08, 76.36, 76.29, 76.13, 76.8, 76.97, 77.0, 76.99, 77.23, 77.18, 77.32, 77.3, 77.2, 77.69, 77.37, 77.8, 77.54, 78.48, 77.28, 77.28, 77.6, 77.92, 77.95, 77.65, 78.12, 77.73, 77.9, 78.51, 77.85, 77.57, 77.94, 78.68, 77.86, 78.18, 78.54, 77.49, 77.52, 77.87, 78.57, 77.86, 77.87, 77.47, 77.97, 78.46, 78.17, 78.15, 78.68, 78.57, 78.59, 78.74, 78.43, 78.63, 77.94, 78.55, 78.67, 78.53, 78.46, 78.73, 78.6, 78.74, 78.8, 79.12, 78.89, 78.84, 79.02, 78.32, 78.56, 78.86, 79.23, 79.12, 79.5, 78.67, 79.07, 79.63, 79.31, 79.74, 79.01, 79.05, 79.05, 79.01, 79.43, 79.09, 78.45, 78.85, 78.86, 79.56, 78.96, 79.48, 79.38, 78.98, 79.32, 79.55, 78.9, 78.57, 78.97, 79.17, 79.41, 79.39, 79.75, 79.09, 79.1, 78.58, 79.31, 79.19, 79.19, 79.62, 78.91, 78.82, 79.17, 79.24, 79.07, 79.1, 79.26, 79.3, 79.63, 79.29, 78.92, 78.85, 79.21, 79.28, 79.38, 79.65, 79.85, 79.75, 79.75, 79.77, 79.99, 79.72, 79.84, 79.53, 79.52, 79.49, 79.96, 79.75, 79.61, 79.67, 79.09, 79.75, 80.1, 79.93, 79.28, 79.34, 79.69, 79.34, 79.63, 80.05, 79.76, 79.81, 79.93, 80.07, 79.89, 79.49, 79.57, 79.44, 79.55, 79.75, 79.33, 79.67, 79.27, 79.33, 79.47, 79.51, 79.62, 80.04, 79.57, 79.63, 79.22, 79.96, 80.02, 79.27, 79.69, 79.65, 79.78, 79.47, 79.57, 79.86, 79.82, 79.83, 79.79, 80.24, 80.2, 79.76, 79.97, 79.97, 79.67, 79.73, 80.31, 79.69, 79.64, 79.34, 79.44, 79.95, 79.93, 79.74, 79.82, 79.77, 79.97, 79.77, 79.98, 79.66, 79.62, 80.08, 80.07, 80.2, 80.0, 79.53, 79.62, 79.29, 79.36, 79.39, 79.62, 79.68, 80.26, 80.06, 79.87, 79.83, 79.68, 79.57, 80.12, 79.5, 80.07, 79.94, 79.65, 79.78, 79.97, 79.76, 79.47, 79.74, 79.8, 79.57, 79.49, 79.64, 79.32]


Y_shift = [36.71, 50.56, 58.71, 64.04, 67.9, 69.3, 69.9, 71.37, 72.24, 73.62, 74.14, 74.38, 74.45, 74.89, 75.4, 75.98, 75.79, 76.49, 76.12, 76.65, 76.7, 76.78, 76.43, 76.93, 77.19, 77.38, 77.08, 76.7, 76.26, 77.11, 77.21, 77.85, 77.66, 77.08, 77.58, 77.05, 77.95, 78.34, 77.88, 77.41, 77.39, 78.04, 78.17, 78.39, 78.57, 78.6, 77.5, 78.07, 78.5, 77.85, 77.94, 78.36, 78.28, 78.42, 78.23, 78.44, 77.76, 77.96, 78.43, 79.01, 78.38, 78.89, 78.94, 77.66, 78.37, 78.55, 78.84, 78.57, 78.18, 78.52, 78.83, 78.72, 79.06, 78.9, 78.74, 79.19, 78.95, 78.91, 78.92, 78.86, 78.55, 78.64, 79.16, 78.77, 78.59, 78.86, 78.89, 78.79, 78.58, 78.6, 78.77, 78.95, 79.51, 79.37, 79.24, 79.09, 78.98, 79.26, 78.42, 79.52, 79.12, 79.3, 78.63, 79.06, 78.9, 79.33, 79.32, 79.36, 79.26, 79.72, 79.8, 79.23, 78.97, 79.03, 79.06, 79.33, 79.49, 79.39, 78.88, 79.04, 79.4, 79.22, 79.22, 79.15, 79.6, 79.34, 80.02, 79.53, 79.37, 79.4, 79.77, 79.95, 79.22, 79.87, 78.75, 78.55, 79.07, 79.67, 79.02, 79.58, 79.34, 79.73, 79.07, 79.28, 79.58, 79.34, 78.94, 79.77, 79.67, 79.63, 79.88, 79.95, 79.75, 79.99, 79.61, 79.7, 79.83, 79.7, 79.87, 79.91, 79.98, 79.28, 80.01, 79.84, 79.59, 78.95, 79.58, 79.83, 79.78, 80.09, 80.02, 80.38, 80.06, 79.96, 79.83, 79.61, 79.7, 79.73, 79.43, 80.16, 79.94, 80.04, 79.71, 79.63, 79.79, 79.81, 79.81, 80.03, 79.82, 79.67, 79.42, 79.57, 79.95, 80.03, 80.17, 79.75, 79.45, 79.35, 79.22, 79.59, 79.76, 79.82, 80.07, 79.74, 79.17, 79.52, 79.55, 79.23, 80.11, 79.78, 79.69, 79.61, 79.47, 79.68, 79.66, 80.26, 79.95, 80.07, 80.14, 80.1, 80.07, 80.15, 80.16, 79.78, 79.93, 80.07, 80.26, 80.54, 79.96, 79.92, 80.35, 80.15, 80.43, 80.6, 80.25, 79.97, 80.15, 80.03, 80.18, 79.85, 79.92, 80.26, 79.9, 80.2, 80.37, 80.32, 79.94, 80.31, 80.2]


X = list(range(len(Y_bas)))

plt.plot(X, Y_bas, color='blue')
plt.plot(X, Y_shift, color='red')
plt.title("en-fr 1000-1000")
plt.show()


# en-fr 10000-100 divmean conf

Y_bas = [31.54, 47.34, 58.97, 63.5, 68.03, 69.65, 71.07, 72.94, 73.63, 73.82, 74.89, 74.84, 75.13, 75.72, 75.61, 76.29, 76.08, 76.36, 76.29, 76.13, 76.8, 76.97, 77.0, 76.99, 77.23, 77.18, 77.32, 77.3, 77.2, 77.69, 77.37, 77.8, 77.54, 78.48, 77.28, 77.28, 77.6, 77.92, 77.95, 77.65, 78.12, 77.73, 77.9, 78.51, 77.85, 77.57, 77.94, 78.68, 77.86, 78.18, 78.54, 77.49, 77.52, 77.87, 78.57, 77.86, 77.87, 77.47, 77.97, 78.46, 78.17, 78.15, 78.68, 78.57, 78.59, 78.74, 78.43, 78.63, 77.94, 78.55, 78.67, 78.53, 78.46, 78.73, 78.6, 78.74, 78.8, 79.12, 78.89, 78.84, 79.02, 78.32, 78.56, 78.86, 79.23, 79.12, 79.5, 78.67, 79.07, 79.63, 79.31, 79.74, 79.01, 79.05, 79.05, 79.01, 79.43, 79.09, 78.45, 78.85, 78.86, 79.56, 78.96, 79.48, 79.38, 78.98, 79.32, 79.55, 78.9, 78.57, 78.97, 79.17, 79.41, 79.39, 79.75, 79.09, 79.1, 78.58, 79.31, 79.19, 79.19, 79.62, 78.91, 78.82, 79.17, 79.24, 79.07, 79.1, 79.26, 79.3, 79.63, 79.29, 78.92, 78.85, 79.21, 79.28, 79.38, 79.65, 79.85, 79.75, 79.75, 79.77, 79.99, 79.72, 79.84, 79.53, 79.52, 79.49, 79.96, 79.75, 79.61, 79.67, 79.09, 79.75, 80.1, 79.93, 79.28, 79.34, 79.69, 79.34, 79.63, 80.05, 79.76, 79.81, 79.93, 80.07, 79.89, 79.49, 79.57, 79.44, 79.55, 79.75, 79.33, 79.67, 79.27, 79.33, 79.47, 79.51, 79.62, 80.04, 79.57, 79.63, 79.22, 79.96, 80.02, 79.27, 79.69, 79.65, 79.78, 79.47, 79.57, 79.86, 79.82, 79.83, 79.79, 80.24, 80.2, 79.76, 79.97, 79.97, 79.67, 79.73, 80.31, 79.69, 79.64, 79.34, 79.44, 79.95, 79.93, 79.74, 79.82, 79.77, 79.97, 79.77, 79.98, 79.66, 79.62, 80.08, 80.07, 80.2, 80.0, 79.53, 79.62, 79.29, 79.36, 79.39, 79.62, 79.68, 80.26, 80.06, 79.87, 79.83, 79.68, 79.57, 80.12, 79.5, 80.07, 79.94, 79.65, 79.78, 79.97, 79.76, 79.47, 79.74, 79.8, 79.57, 79.49, 79.64, 79.32]


Y_divmean = [47.38, 51.17, 55.03, 57.07, 58.47, 58.75, 60.39, 61.24, 61.22, 61.86, 62.43, 61.52, 61.94, 62.79, 61.5, 63.9, 59.56, 62.3, 62.42, 62.61, 63.17, 63.24, 63.66, 63.41, 61.24, 63.37, 63.42, 62.82, 64.03, 64.2, 64.37, 63.38, 63.31, 62.99, 63.65, 62.99, 63.82, 64.58, 63.2, 63.89, 64.27, 63.48, 65.12, 63.93, 64.17, 64.22, 64.17, 64.24, 64.06, 64.33, 65.05, 64.6, 65.26, 64.98, 65.76, 65.58, 65.85]


X = list(range(len(Y_bas)))

plt.plot(X, Y_bas[:len(X)], color='blue')
plt.plot(X, Y_shift, color='red')
plt.title("en-fr 10000-100 divmean conf")
plt.show()


# de-fr 1000-1 divmean 120 epochs batch size = 4

Y_bas = [7.36, 7.05, 6.32, 10.14, 8.01, 10.18, 8.14, 10.48, 7.74, 10.82, 10.88, 12.35, 9.91, 11.36, 10.62, 9.5, 12.34, 11.53, 12.75, 10.6, 8.89, 12.5, 9.39, 9.93, 10.12, 9.35, 11.13, 9.46, 12.31, 11.05, 10.52, 11.39, 11.94, 11.45, 12.37, 12.43, 12.54, 12.3, 11.68, 11.86, 12.87, 13.54, 13.81, 12.08, 11.01, 12.21, 12.04, 10.97, 10.52, 11.37, 11.18, 10.97, 9.93, 11.28, 11.11, 11.37, 11.86, 11.55, 10.82, 11.03, 10.93, 10.16, 10.79, 11.2, 11.28, 11.6, 10.52, 10.89, 11.36, 10.41, 9.3, 10.17, 10.07, 9.99, 10.16, 10.55, 10.46, 11.66, 10.15, 12.73, 13.52, 13.16, 12.33, 13.28, 13.02, 12.4, 10.67, 10.78, 10.52, 10.7, 10.44, 10.27, 10.75, 10.46, 10.77, 11.36, 11.88, 11.66, 11.46, 10.91, 11.75, 11.46, 11.4, 10.8, 11.17, 11.15, 11.14, 11.8, 11.84, 11.55, 11.57, 11.61, 10.49, 11.31, 12.38, 12.09, 11.89, 11.75, 11.81]

Y_divmean = [6.49, 7.36, 9.13, 8.21, 10.06, 9.0, 8.38, 9.11, 11.45, 12.35, 15.06, 11.35, 13.35, 13.6, 13.77, 16.29, 12.62, 17.1, 13.69, 12.81, 14.14, 12.54, 12.63, 16.72, 15.08, 12.32, 12.11, 12.63, 12.61, 13.01, 12.14, 11.97, 10.52, 12.15, 11.54, 14.17, 11.16, 11.07, 10.54, 11.59, 12.08, 11.69, 11.88, 14.14, 14.15, 14.51, 13.07, 14.04, 13.28, 11.34, 14.08, 12.86, 12.72, 13.92, 13.39, 13.83, 12.62, 12.91, 12.25, 11.33, 12.73, 12.37, 12.56, 13.99, 15.17, 14.85, 14.46, 15.19, 15.33, 15.32, 15.34, 17.07, 18.39, 17.4, 16.2, 17.39, 15.75, 15.48, 16.07, 15.42, 14.96, 14.29, 14.99, 13.99, 16.15, 15.03, 13.58, 13.29, 13.52, 14.67, 14.8, 14.82, 15.23, 16.16, 16.17, 17.53, 16.37, 17.56, 18.07, 16.42, 17.91, 17.25, 18.52, 17.39, 18.9, 19.25, 18.14, 17.83, 16.99, 18.05, 18.08, 17.8, 17.89, 18.01, 14.35, 15.44, 14.72, 15.93, 16.3]

X = list(range(len(Y_bas)))

plt.plot(X, Y_bas[:len(X)], color='blue', label='baseline')
plt.plot(X, Y_divmean, color='red', label='with confidence')
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("LAS accuracy")
plt.title("de-fr 1000-0")
plt.show()

# de-fr 1000-1 divmean  testing on German 120 epochs batch size = 4

Y_bas = [38.24, 47.05, 50.16, 52.1, 52.77, 53.59, 56.64, 57.18, 56.69, 57.59, 57.7, 59.36, 57.97, 57.29, 58.81, 60.01, 60.69, 59.02, 59.89, 59.42, 59.7, 60.24, 59.23, 60.23, 59.34, 60.05, 59.47, 60.14, 60.63, 60.0, 60.39, 59.72, 60.64, 61.64, 60.26, 61.16, 61.39, 60.62, 60.0, 61.36, 61.27, 61.65, 60.78, 60.93, 61.45, 61.34, 61.38, 61.54, 61.72, 62.11, 61.87, 61.92, 61.9, 61.52, 61.97, 61.67, 61.73, 61.49, 61.61, 62.03, 62.27, 62.57, 61.74, 60.63, 61.78, 60.95, 61.13, 62.13,
62.1, 61.9, 62.27, 61.64, 62.07, 61.95, 62.43, 61.74, 61.92, 61.54, 61.8, 61.83, 61.64, 62.01, 62.05, 62.07, 62.14, 62.44, 62.5, 61.9, 62.46, 63.33, 63.15, 62.82, 62.86, 63.53, 63.29, 62.79, 62.69, 63.06, 63.09, 62.66, 62.75, 62.77, 62.63, 62.73, 62.64, 62.97, 63.52, 63.52, 62.73, 63.22, 63.42, 63.49, 63.39, 63.29, 63.13, 63.07, 63.23, 63.45, 63.47]

Y_divmean = [39.39, 43.93, 44.64, 50.97, 51.78, 52.76, 54.42, 56.3, 56.79, 56.25, 56.75, 57.73, 57.27, 57.61, 57.74, 57.92, 59.01, 58.39, 59.07, 59.35, 58.72, 58.69, 58.71, 58.09, 59.32, 58.72, 58.32, 59.44, 59.98, 59.56, 59.3, 59.71, 59.66, 59.23, 59.43, 59.43, 59.82, 60.26, 60.39, 60.15, 60.64, 60.29, 59.73, 60.2, 59.9, 59.81, 60.27, 60.13, 60.58, 60.06, 60.79, 60.64, 60.73, 60.7, 60.93, 61.24, 60.73, 61.52, 61.33, 61.4, 60.81, 60.46, 60.5, 60.58, 60.74, 60.91, 60.29, 60.67, 61.07, 60.83, 60.83, 60.6, 61.69, 61.06, 61.98, 61.8, 61.29, 61.03, 61.18, 61.5, 61.53, 61.48, 61.65, 61.17, 61.21, 61.29, 60.86, 61.04, 61.21, 61.18, 61.33, 61.15, 61.44, 61.71, 61.58, 61.95, 61.81, 61.52, 62.24, 61.72, 61.7, 61.86, 62.09, 62.17, 62.36, 61.89, 61.75, 62.09, 61.99, 61.98, 61.73, 61.58, 62.06, 62.24, 61.8, 62.7, 62.6, 62.55, 62.71]


X = list(range(len(Y_bas)))

plt.plot(X, Y_bas[:len(X)], color='blue', label='baseline')
plt.plot(X, Y_divmean, color='red', label="with confidence")
plt.legend()
plt.title("de-fr 1000-0")
plt.xlabel("Number of epochs")
plt.ylabel("LAS accuracy")
plt.show()


# de-fr 1000-1 divmean 200 epochs batch size = 16
Y_bas = [7.08, 10.87, 10.1, 9.85, 10.76, 10.14, 11.17, 9.24, 9.81, 10.02, 7.29, 8.67, 10.19, 12.26, 13.88, 13.28, 9.84, 8.33, 10.95, 10.0, 11.23, 10.87, 9.52, 10.34, 11.76, 10.24, 9.92, 9.09, 9.79, 11.62, 13.07, 10.61, 11.23, 10.74, 10.59, 15.53, 13.0, 11.62, 12.17, 13.65, 13.41, 14.14, 12.45, 12.59, 11.88, 12.06, 12.07, 11.01, 13.67, 11.94, 13.94, 10.84, 12.63, 10.2, 11.75, 11.19, 13.12, 12.81, 11.6, 12.11, 10.69, 10.73, 11.93, 11.77, 11.32, 13.19, 12.84, 11.91, 12.76, 13.09, 12.52, 12.97, 12.27, 11.52, 11.64, 11.56, 11.24, 11.41, 10.01, 13.91, 9.97, 11.6, 10.9, 10.28, 9.83, 9.99, 10.72, 10.85, 10.6, 11.32, 10.15, 14.47, 11.45, 13.17, 12.45, 11.25, 11.38, 11.12, 12.7, 12.43, 13.41, 12.78, 13.43, 13.91, 13.03, 13.78, 13.05, 12.75, 12.92, 11.79, 12.08, 13.3, 15.75, 15.5, 10.29, 11.6, 11.45, 12.01, 11.51, 12.6, 12.48, 12.81, 12.26, 12.29, 12.79, 12.12, 11.94, 11.81, 11.71, 11.46, 11.51, 11.83, 11.91, 12.03, 13.01, 12.99, 13.71, 13.04, 12.72, 13.34, 12.26, 11.25, 10.91, 12.69, 12.49, 11.61, 12.07, 11.87, 13.0, 12.65, 13.78, 13.54, 13.52, 13.13, 12.57, 12.49, 12.79, 12.02, 11.89, 12.32, 13.5, 12.45, 13.86, 13.1, 13.34, 13.25, 14.73, 14.22, 14.65, 12.65, 14.32, 14.39, 13.31, 13.8, 13.32, 13.13, 12.19, 13.42, 12.55, 14.29, 13.17, 12.16, 12.04, 13.0, 12.32, 12.2, 11.86, 11.88, 11.42, 11.99, 13.1, 13.79, 14.23, 14.55, 14.3, 14.29, 13.9, 14.88, 14.78]


Y_divmean = [7.8, 7.0, 8.21, 9.16, 8.18, 11.29, 8.47, 6.97, 8.11, 10.57, 8.97, 8.07, 9.57, 9.27, 10.34, 11.1, 10.21, 10.65, 7.73, 11.17, 9.52, 10.25, 11.1, 10.08, 9.2, 9.88, 11.04, 10.22, 9.04, 10.06, 10.65, 10.34, 10.36, 11.43, 10.71, 9.59, 9.22, 8.18, 11.75, 10.6, 11.03, 9.55, 9.81, 10.55, 11.2, 9.58, 10.35, 11.56, 10.13, 11.25, 10.92, 10.54, 9.7, 10.1, 10.87, 9.53, 10.41, 9.18, 9.65, 12.33, 11.98, 10.89, 10.97, 10.92, 11.21, 11.25, 10.58, 11.95, 12.14, 10.46, 11.19, 10.56, 10.23, 9.89, 10.35, 10.23, 9.71, 10.43, 10.12, 9.81, 10.75, 10.74, 9.65, 10.12, 9.81, 9.1, 10.48, 9.99, 9.48, 10.52, 9.46, 10.24, 9.87, 9.59, 10.84, 10.3, 9.23, 10.07, 9.77, 9.3, 9.25, 9.44, 10.59, 8.76, 8.75, 9.2, 9.95, 10.42, 11.43, 11.25, 10.47, 10.72, 11.14, 10.22, 9.94, 12.78, 11.79, 11.02, 11.93, 10.44, 10.3, 9.81, 10.38, 12.73, 10.1, 11.11, 10.45, 12.61, 11.57, 10.3, 11.14, 12.58, 12.88, 13.51, 12.81, 11.34, 12.33, 11.8, 11.75, 10.27, 11.12, 10.33, 10.64, 10.76, 9.45, 10.22, 11.5, 11.48, 10.91, 11.51, 10.02, 10.07, 11.56, 11.25, 10.66, 10.13, 10.04, 10.66, 10.63, 10.12, 11.44, 10.99, 13.14, 13.21, 12.23, 12.09, 12.54, 13.53, 14.5, 13.43, 13.5, 12.73, 12.81, 12.76, 12.66, 13.17, 13.4, 13.51, 12.63, 13.17, 12.66, 11.83, 11.98, 11.59, 12.11, 12.57, 11.89, 12.21, 12.13, 12.81, 11.91, 12.07, 11.48, 12.16, 12.12, 11.21, 10.24, 10.76, 11.85]


X = list(range(len(Y_bas)))

plt.plot(X, Y_bas[:len(X)], color='blue')
plt.plot(X, Y_divmean, color='red')
plt.title("de-fr 1000-1 divmean 200 epochs batch size = 16")
plt.show()


# fi-se 1000-1 divmean 180 epochs bs = 2
Y_bas = [7.73, 9.35, 7.46, 7.05, 9.24, 8.57, 9.66, 8.4, 7.99, 8.6, 8.11, 9.74, 9.1, 9.02, 8.95, 9.35, 9.79, 9.83, 9.85, 9.77, 9.29, 9.31, 9.41, 8.73, 8.61, 8.47, 7.93, 8.95, 9.64, 9.35, 9.1, 8.96, 9.36, 9.83, 9.05, 8.46, 8.59, 9.19, 9.22, 8.83, 9.46, 8.82, 9.12, 9.09, 9.89, 9.17, 9.39, 9.69, 9.74, 9.73, 9.03, 9.41, 9.08, 9.24, 9.69, 9.09, 9.29, 9.21, 9.75, 9.32, 9.66, 9.87, 9.43, 9.21, 9.02, 9.25, 9.32, 9.63, 9.08, 9.22, 9.04, 9.22, 9.1, 8.85, 8.96, 8.73, 9.23, 9.18, 9.33, 9.25, 9.27, 9.08, 9.28, 9.13, 9.16, 9.23, 9.34, 9.05, 9.09, 9.28, 9.14, 9.36, 9.44, 9.45, 9.41, 9.55, 9.26, 9.15, 9.4, 9.29, 9.2, 9.67, 9.78, 9.63, 9.49, 9.52, 9.31, 9.18, 9.39, 9.4, 9.45, 9.52, 9.51, 9.64, 9.68, 9.54, 9.81, 9.56, 9.56, 9.36, 9.49, 9.47, 9.45, 9.26, 9.39, 9.52, 9.69, 9.55, 9.5, 9.58, 9.53, 9.46, 9.57, 9.48, 9.51, 9.61, 9.54, 9.52, 9.47, 9.64, 9.6, 9.55, 9.56, 9.48, 9.52, 9.62, 9.61, 9.59, 9.64, 9.77, 9.71, 9.63, 9.59, 9.65, 9.56, 9.7, 9.73, 9.73, 9.71, 9.68, 9.73, 9.8, 9.77, 9.76, 9.73, 9.66, 9.76, 9.82, 9.86, 9.81, 9.72, 9.74, 9.74, 9.68, 9.67, 9.67, 9.76, 9.72, 9.73]

Y_divmean = [8.61, 8.06, 9.02, 10.22, 8.88, 10.02, 9.28, 9.2, 9.56, 9.0, 9.72, 9.25, 8.94, 9.25, 8.57, 9.11, 9.05, 9.23, 9.66, 8.89, 8.74, 9.11, 9.07, 9.1, 9.04, 8.23, 9.16, 8.61, 9.03, 8.53, 8.74, 9.34, 8.84, 9.2, 8.43, 8.49, 8.48, 9.24, 9.28, 9.46, 9.29, 8.72, 8.84, 8.92, 9.03, 9.02, 9.3, 9.08, 9.2, 9.15, 8.64, 8.7, 8.15, 9.06, 8.97, 8.52, 9.36, 8.87, 8.79, 8.75, 8.7, 8.81, 8.73, 9.09, 9.03, 9.29, 9.14, 9.16, 8.86, 9.19, 8.99, 9.04, 9.29, 9.31, 9.02, 8.64, 9.11, 8.63, 8.86, 8.93, 8.95, 9.18, 9.12, 9.28, 8.91, 9.19, 8.82, 9.1, 9.08, 9.03, 8.82, 8.96, 8.75, 8.94, 9.12, 8.74, 8.48, 8.68, 9.28, 9.29, 9.35, 9.38, 9.15, 9.12, 9.25, 9.24, 9.32, 9.23, 9.26, 9.29, 9.17, 9.19, 9.2, 9.08, 9.4, 9.17, 9.01, 8.99, 9.2, 9.02, 9.19, 9.0, 9.19, 9.08, 9.07, 9.06, 9.07, 9.05, 9.26, 9.25, 9.32, 9.35, 9.37, 9.25, 9.22, 9.36, 9.62, 9.31, 9.24, 9.09, 9.18, 9.15, 9.15, 9.25, 9.34, 9.3, 9.2, 9.3, 9.32, 9.36, 9.43, 9.3, 9.31, 9.37, 9.35, 9.4, 9.34, 9.36, 9.39, 9.39,
9.4, 9.36, 9.33, 9.33, 9.24, 9.21, 9.2, 9.27, 9.41, 9.29, 9.37, 9.22, 9.25, 9.31, 9.29, 9.39, 9.28, 9.31, 9.31]

X = list(range(len(Y_bas)))

plt.plot(X, Y_bas[:len(X)], color='blue')
plt.plot(X, Y_divmean, color='red')
plt.title("fi-se 1000-1 divmean 180 epochs bs = 2")
plt.show()


# fi-se 1000-1 divmean 180 epochs bs = 4
Y_bas = [9.57, 8.48, 7.64, 9.87, 7.09, 7.42, 8.41, 8.92, 8.35, 8.2, 9.04, 8.05, 9.76, 9.13, 8.43, 9.48, 9.01, 9.05, 9.08, 9.39, 9.18, 9.21, 8.78, 9.52, 8.47, 9.9, 8.21, 8.96, 9.42, 9.95, 10.25, 9.49, 9.91, 10.38, 9.91, 9.16, 9.71, 8.91, 9.07, 9.58, 8.47, 9.38, 8.7, 9.43, 8.48, 9.83, 8.59, 9.51, 9.84, 8.52, 8.98, 9.1, 8.9, 8.76, 9.2, 8.92, 8.35, 8.63, 9.85, 10.13, 9.43, 9.42, 9.5, 9.58, 9.69, 9.85, 9.09, 9.17, 9.23, 9.18, 9.4, 9.5, 8.99, 8.93, 8.95, 9.1, 9.57, 9.01, 9.12, 9.14, 9.4, 9.21, 9.3, 8.86, 8.95, 8.8, 9.1, 9.34, 9.47, 9.42, 9.53, 8.66, 8.72, 9.68, 8.49, 9.81, 9.96, 9.81, 9.82, 9.81, 9.52, 9.52, 9.3, 9.17, 9.92, 9.97, 9.32, 9.3, 9.34, 9.08, 9.45, 9.71, 9.71, 9.65, 9.7, 9.41, 9.36, 9.53, 9.67, 9.58, 8.98, 9.1, 9.04, 8.9, 9.07, 8.87, 9.05, 8.96, 9.07, 9.85, 9.62, 9.56, 9.75, 9.37, 9.93, 9.39, 9.21, 9.03, 9.01, 9.29, 9.33, 9.75, 9.61, 9.52, 9.58, 9.26, 9.53, 9.37, 8.96, 9.05, 9.13, 9.1, 9.18, 9.17, 9.69, 9.59, 9.62, 9.59, 9.73, 9.61, 9.37, 9.27, 9.42, 9.49, 9.67, 9.51, 9.4, 9.64, 9.5, 9.49, 9.67, 9.62, 9.29, 9.33, 9.34, 9.09, 9.14, 9.4, 9.29]

Y_divmean = [9.22, 10.74, 12.02, 11.02, 11.71, 10.59, 10.84, 11.11, 11.49, 11.21, 11.07, 11.29, 10.74, 9.97, 9.72, 10.62, 10.4, 10.84, 10.44, 9.97, 10.17, 10.25, 9.66, 8.79, 9.78, 10.53, 8.9, 10.57, 9.84, 9.66, 9.4, 8.41, 8.7, 9.51, 9.87, 9.74, 9.69, 10.18, 10.32, 10.35, 9.7, 10.32, 10.58, 9.3, 9.78, 9.7, 10.09, 10.34, 9.84, 10.0, 10.02, 10.11, 9.7, 10.01, 10.1, 10.39, 10.39, 10.66, 10.0, 10.12, 9.96, 9.81, 9.98, 9.67, 10.21, 9.43, 10.04, 9.7, 10.33, 10.08, 9.93, 9.78, 9.9, 10.22, 9.52, 9.67, 9.52, 9.49, 10.28, 10.42, 10.11, 9.79, 9.77, 9.7, 9.37, 9.27, 9.97, 10.17, 10.02, 10.18, 10.11, 10.4, 9.37, 10.16, 10.02, 9.46, 9.49, 9.87, 9.8, 9.6, 9.85, 9.87, 10.17, 9.53, 10.16, 9.83, 9.62, 9.85, 10.24, 10.22, 9.88, 9.59, 9.32, 10.18, 10.7, 10.54, 10.51, 9.9, 10.14, 10.02, 10.27, 10.19, 10.46, 10.2, 10.02, 9.8, 9.72, 9.96, 10.06, 9.95, 10.16, 10.18, 10.02, 10.0, 10.06, 10.05, 10.29, 9.95, 10.1, 9.83, 10.19, 10.56, 10.67, 10.52, 10.79, 10.77, 10.75, 10.65, 10.61, 10.58, 10.79, 10.53, 10.83, 10.29, 10.7, 10.81, 10.61, 10.3, 10.74, 10.86, 10.93, 10.69, 10.7, 10.41, 10.45, 10.52, 10.23, 10.35, 10.39, 10.26, 10.53, 10.38, 10.46, 10.28, 10.26, 10.39, 10.38, 10.4, 10.49]


X = list(range(len(Y_bas)))

plt.plot(X, Y_bas[:len(X)], color='blue')
plt.plot(X, Y_divmean, color='red')
plt.title("fi-se 1000-1 divmean 180 epochs bs = 4")
plt.show()


# fi-se 1000-10 divmean 180 epochs bs = 2
Y_bas = [10.44, 9.26, 11.01, 9.1, 11.47, 12.06, 10.75, 12.7, 12.63, 12.86, 12.85, 12.2, 13.0, 13.2, 13.05, 12.4, 13.49, 12.39, 13.55, 13.34, 13.6, 13.96, 12.92, 14.22, 13.0, 14.87, 14.1, 14.49, 14.04, 13.19, 13.77, 13.58, 14.09, 13.48, 13.76, 14.02, 13.37, 13.25, 13.57, 14.6, 14.04, 14.13, 15.11, 14.61, 15.16, 13.24, 13.92, 15.43, 14.38, 15.07, 16.03, 14.9, 14.19, 14.8, 14.62, 14.81, 14.55, 15.26, 14.79, 13.48, 14.58, 13.91, 13.9, 14.08, 14.15, 14.6, 14.05, 14.9, 14.62, 15.32, 14.87, 15.02, 14.86, 14.84, 15.09, 14.77, 15.66, 15.33, 15.16, 15.42, 14.4, 14.04, 14.69, 14.68, 14.41, 14.49, 14.54, 15.0, 14.68, 14.83, 14.76, 14.72, 15.31, 15.11, 15.17, 15.0, 14.47, 14.62, 14.61, 14.57, 14.33, 14.36, 14.71, 15.35, 15.41, 14.87, 14.91, 15.04, 15.28, 14.97, 15.22, 15.15, 15.09, 14.37, 14.16, 14.57, 14.45, 15.0, 14.59, 14.96, 14.56, 14.36, 14.55, 14.23, 13.99, 14.24, 14.55, 14.93, 14.11, 14.29, 14.56, 14.26, 14.64, 14.5, 14.67, 14.74, 14.83, 14.85, 14.57, 14.53, 14.66, 14.58, 14.37, 14.62, 14.47, 14.48, 14.36, 14.52, 14.89, 14.88, 14.91, 15.01, 15.06, 15.02, 14.83, 14.81, 14.82, 14.97, 15.21, 14.91, 14.95, 15.13, 15.06, 15.26, 15.19, 14.98, 15.11, 15.01, 15.15, 15.24, 14.9, 14.91, 15.1, 14.93, 14.77, 14.74, 14.83, 14.97, 14.94]

Y_divmean = [8.39, 12.09, 12.69, 12.86, 11.79, 12.64, 13.61, 13.48, 12.97, 12.04, 11.24, 13.14, 13.66, 13.26, 12.38, 12.48, 12.73, 12.34, 12.15, 12.29, 12.79, 12.98, 14.6, 13.03, 13.19, 13.1, 12.81, 13.36, 13.0, 13.05, 12.46, 12.79, 12.66, 13.34, 13.3, 13.26, 13.32, 12.85, 13.2, 13.76, 13.25, 13.49, 13.74, 13.4, 13.36, 14.36, 14.34, 14.0, 14.36, 13.39, 14.06, 13.84, 13.48, 14.36, 13.57, 13.74, 14.16, 13.68, 13.61, 14.1, 14.46, 13.87, 13.25, 13.57, 13.34, 13.09, 14.37, 12.92, 12.89, 12.72, 13.94, 12.94, 13.81, 13.62, 13.6, 13.61, 14.19, 14.24, 14.03, 13.97, 13.45, 14.32, 14.29, 14.38, 14.64, 14.37, 13.92, 13.66, 13.83, 13.38, 13.66, 13.76, 13.26, 12.97, 13.12, 13.55, 13.3, 13.54, 13.38, 13.76, 13.66, 13.92, 13.7, 13.97, 13.89, 13.94, 14.04, 14.32, 14.46, 14.05, 13.83, 14.47, 13.97, 13.98, 13.57, 13.48, 14.18, 13.84, 13.65, 13.59, 13.72, 13.76, 13.84, 14.06, 13.83, 13.46, 13.93, 13.51, 13.58, 13.61, 14.15, 14.29, 14.27, 14.41, 14.3, 14.8, 13.8, 14.13, 13.95, 13.75, 13.94, 13.86, 14.01, 14.26, 14.07, 14.23, 14.38, 14.48, 14.38, 14.22, 14.23, 14.19, 14.29, 14.2, 14.34, 14.11, 14.39, 14.35, 14.67, 14.44, 14.31, 14.33, 14.45, 14.5, 14.68, 14.54, 14.62, 14.61, 14.46, 14.72, 14.53, 14.45, 14.36, 14.7, 14.58, 14.58, 14.18, 14.54, 14.44]


X = list(range(len(Y_bas)))

plt.plot(X, Y_bas[:len(X)], color='blue')
plt.plot(X, Y_divmean, color='red')
plt.title("fi-se 1000-10 divmean 180 epochs bs = 2")
plt.show()

# fi-se 1000-10 noconf 180 epochs bs = 4 levenshtein embeddings
Y_bas = [10.17, 11.61, 10.29, 11.8, 11.77, 12.63, 13.87, 12.64, 12.05, 11.12, 11.8, 12.82, 11.63, 12.5, 12.6, 13.94, 12.6, 12.59, 14.08, 12.53, 13.21, 13.39, 12.9, 12.69, 13.06, 12.27, 13.42, 13.97, 13.15, 13.81, 13.84, 13.73, 14.28, 12.71, 14.73, 14.06, 14.08, 14.37, 14.0, 14.6, 14.29, 13.68, 14.12, 13.59, 13.55, 14.81, 14.62, 13.98, 14.1, 13.86, 13.82, 14.21, 14.5, 13.51, 13.27, 13.78, 14.19, 13.71, 13.73, 13.71, 14.81, 13.98, 14.81, 14.69, 14.13, 14.63, 14.64, 14.67, 14.95, 14.93, 14.53, 14.69, 14.8, 15.18, 15.06, 14.48, 14.48, 15.43, 15.06, 15.39, 15.53, 14.63, 15.29, 15.2, 15.18, 15.45, 15.08, 14.37, 14.95, 14.93, 15.09, 15.09, 15.13, 14.76, 14.67, 14.97, 15.03, 14.75, 14.34, 14.58, 14.63, 14.97, 15.12, 15.15, 14.99, 14.85, 14.69, 14.39, 14.94, 15.0, 14.25, 14.05, 14.12, 14.64, 14.62, 14.95, 15.0, 14.85, 15.0, 14.87, 14.55, 14.59, 14.81, 14.64, 14.55, 14.74, 14.79, 14.75, 14.65, 15.07, 15.19, 15.04, 15.13, 15.19, 15.28, 14.94, 15.03, 15.26, 15.03, 14.93, 14.79, 14.78, 14.73, 14.37, 14.43, 14.51, 14.64, 15.03, 15.32, 14.98, 14.57, 14.5, 14.58, 14.61, 15.0, 14.66, 14.64, 14.58, 14.74, 14.52, 14.55, 14.34, 14.08, 14.21, 14.07, 14.14, 14.32, 14.44, 14.31, 14.37, 14.33, 14.57, 14.65, 14.64, 14.75, 14.53, 14.72, 14.76, 14.89]
Y_leven = [10.05, 12.36, 11.46, 11.89, 10.6, 12.69, 12.09, 11.99, 11.54, 10.55, 11.95, 12.24, 12.02, 11.74, 12.31, 13.21, 13.98, 11.45, 12.97, 12.26, 11.85, 12.93, 11.63, 11.23, 11.78, 11.72, 11.93, 12.69, 11.86, 12.32, 12.46, 13.39, 13.8, 13.19, 12.66, 13.72, 13.16, 14.04, 12.54, 13.57, 13.48, 13.23, 13.94, 13.3, 13.82, 13.99, 14.09, 13.28, 13.54, 13.36, 13.71, 13.86, 14.47, 13.05, 13.15, 13.7, 12.67, 13.34, 12.53, 13.47, 13.79, 13.73, 13.62, 13.44, 14.1, 14.3, 14.22, 14.16, 13.77, 13.71, 13.65, 13.38, 13.66, 13.67, 13.07, 12.32, 13.24, 13.31, 12.79, 12.83, 13.32, 13.62, 12.7, 12.93, 12.56, 12.93, 13.17, 13.32, 13.24, 13.54, 12.9, 13.27, 12.94, 13.67, 13.31, 13.19, 13.85, 13.53, 13.73, 13.6, 13.54, 13.38, 13.24, 13.38, 13.35, 13.35, 13.58, 13.63, 13.88, 13.65, 13.88, 13.54, 13.98, 14.15, 14.21, 14.29, 14.17, 13.94, 13.61, 13.35, 13.68, 13.69, 13.92, 13.91, 13.41, 13.59, 13.77, 13.69, 13.38, 13.64, 13.53, 14.18, 13.92, 13.98, 13.95, 13.88, 13.68, 13.95, 13.91, 14.26, 14.02, 14.16, 13.87, 14.01, 13.91, 13.88, 13.97, 14.22, 14.15, 13.92, 13.78, 14.28, 14.06, 13.86, 13.99, 14.38, 14.01, 14.09, 14.24, 14.09, 14.23, 14.09, 14.2, 14.14, 13.95, 14.08, 13.93, 14.22, 14.23, 14.06, 14.09, 14.0, 13.9, 14.27, 14.17, 14.27, 14.22, 14.07, 14.23]


X = list(range(len(Y_bas)))

plt.plot(X, Y_bas[:len(X)], color='blue')
plt.plot(X, Y_leven, color='red')
plt.title("fi-se 1000-10 noconf 180 epochs bs = 4 levenshtein embeddings")
plt.show()

# fi-se entire dataset 180 epochs bs = 32 levenshtein embeddings

Y_bas = [48.06, 55.4, 57.57, 59.98, 61.33, 63.62, 63.45, 64.09, 65.33, 64.97, 65.8, 65.87, 66.81, 66.44, 67.65, 67.73, 67.73, 67.43, 67.71, 68.02, 68.32, 67.76, 68.94, 68.43, 68.63, 67.92, 68.38, 68.77, 68.71, 68.57, 68.74, 69.02, 69.17, 69.02, 69.34, 68.82, 68.35, 68.77, 68.91, 69.25, 68.88, 68.81, 68.74, 68.9, 69.03, 68.89, 68.68, 69.19, 69.14, 68.91, 69.07, 68.71, 69.27, 69.27, 68.98, 68.77, 69.26, 69.39, 69.26, 69.14, 69.33, 69.29, 69.38, 69.5, 69.56, 69.47, 69.02, 69.14, 69.15, 69.2, 69.09, 69.44, 69.26, 69.33, 69.14, 69.2, 69.2, 69.26, 69.35, 69.53, 69.48, 69.42, 69.32, 69.53, 69.47, 69.32, 69.23, 69.34, 69.45, 69.71, 69.67, 69.46, 69.52, 69.51, 69.41, 69.49, 69.64, 69.41, 69.32, 69.28, 69.39, 69.49, 69.34, 69.34, 69.42, 69.41, 69.5, 69.66, 69.65, 69.63, 69.56, 69.53, 69.6, 69.59, 69.65, 69.56, 69.49, 69.55, 69.49, 69.43, 69.48, 69.49, 69.43, 69.42, 69.39, 69.44, 69.39, 69.37, 69.35, 69.33, 69.36, 69.44, 69.44, 69.39, 69.41, 69.44, 69.42, 69.41, 69.36, 69.37, 69.35, 69.37, 69.38, 69.38, 69.38, 69.38, 69.38, 69.4, 69.42, 69.39, 69.39, 69.41, 69.38, 69.36, 69.37, 69.37, 69.38, 69.39, 69.4, 69.4, 69.4, 69.4, 69.39, 69.36, 69.39, 69.39, 69.39, 69.39, 69.39, 69.36, 69.39, 69.38, 69.39, 69.38, 69.39, 69.39, 69.39, 69.4, 69.4]

Y_leven = [51.35, 55.98, 60.0, 61.39, 61.59, 63.77, 63.9, 64.71, 64.95, 65.41, 65.92, 65.36, 67.02, 66.01, 66.58, 67.08, 67.44, 67.0, 67.52, 67.53, 68.57, 68.55, 68.9, 68.36, 68.5, 68.91, 69.26, 69.3, 69.67, 69.69, 68.52, 68.6, 69.71, 67.79, 69.26, 69.25, 69.96, 70.27, 70.64, 70.35, 70.02, 69.92, 69.81, 69.9, 69.58, 69.9, 69.5, 70.26, 70.12, 70.68, 71.08, 70.25, 69.87, 70.51, 71.14, 70.88, 70.86, 70.57, 70.48, 70.12, 70.91, 70.71, 70.53, 70.04, 70.87, 71.07, 70.76, 71.03, 70.65, 71.14, 70.6, 70.23, 70.64, 71.05, 70.5, 71.1, 70.54, 71.11, 70.71, 70.35, 70.24, 70.71, 70.97, 70.73, 70.89, 71.29, 71.07, 71.38, 70.81, 71.1, 71.14, 70.93, 70.29, 70.73, 70.76, 70.95, 70.9, 71.04, 71.22, 71.05, 70.94, 70.97, 71.21, 71.17, 70.8, 71.17, 71.17, 70.88, 70.78, 71.16, 71.0, 71.05, 71.28, 71.45, 71.05, 71.02, 71.09, 71.04, 70.82, 71.08, 71.11, 71.23, 70.96, 71.01, 71.05, 71.1, 71.08, 71.16, 71.14, 71.29, 71.21, 71.52, 71.6, 71.45, 71.35, 71.52, 71.5, 71.21, 71.26, 71.23, 71.57, 71.44, 71.43, 71.27, 71.33, 71.49, 71.56, 71.5, 71.32, 71.29, 71.33, 71.37, 71.52, 71.52, 71.65, 71.54, 71.41, 71.58, 71.42, 71.41, 71.45, 71.47, 71.45, 71.4, 71.38, 71.41, 71.51, 71.23, 71.35, 71.27, 71.42, 71.4, 71.41, 71.41, 71.44, 71.35, 71.38, 71.34, 71.49]


X = list(range(len(Y_bas)))

plt.plot(X, Y_bas[:len(X)], color='blue')
plt.plot(X, Y_leven, color='red')
plt.title("fi-se entire dataset 180 epochs bs = 32 levenshtein embeddings")
plt.show()
