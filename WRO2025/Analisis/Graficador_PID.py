import matplotlib.pyplot as plt

# Ruta al archivo
ruta_archivo = "C:/Users/pepeb/OneDrive/Documentos/GitHub/NanoRob_Hadrones/WRO2025/LecturasPID.txt"

# Leer todas las líneas no vacías
with open(ruta_archivo, "r") as archivo:
    lineas = [linea.strip() for linea in archivo if linea.strip()]

# Extraer parámetros al final del archivo
Hz = float(lineas[-1])
kd = round(float(lineas[-2]), 3)
ki = round(float(lineas[-3]), 3)
kp = round(float(lineas[-4]), 3)
velocidad = float(lineas[-5])

# Las demás son lecturas: error;corrección
lecturas = lineas[:-5]

errores = []
correcciones = []

for linea in lecturas:
    try:
        error_str, correccion_str = linea.split(";")
        errores.append(float(error_str))
        correcciones.append(float(correccion_str))
    except ValueError:
        print(f"Línea ignorada por formato incorrecto: {linea}")

# Crear eje de tiempo
tiempos = [i / Hz for i in range(len(errores))]

# -------------------------------
# GRÁFICA 1: Error vs Tiempo
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(tiempos, errores, label="Error PID", color="tab:blue")
plt.xlabel("Tiempo (s)")
plt.ylabel("Error")
plt.title(f"Error PID vs Tiempo\nvel={velocidad}, kp={kp}, ki={ki}, kd={kd}, Frecuencia={Hz} Hz")
plt.ylim(-150, 150)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("grafica_error.png", dpi=300)

# -------------------------------
# GRÁFICA 2: Corrección vs Tiempo
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(tiempos, correcciones, label="Corrección PID", color="tab:green")
plt.xlabel("Tiempo (s)")
plt.ylabel("Corrección")
plt.title(f"Corrección PID vs Tiempo\nvel={velocidad}, kp={kp}, ki={ki}, kd={kd}, Frecuencia={Hz} Hz")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("grafica_correccion.png", dpi=300)
