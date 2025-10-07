import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# -------- CONFIGURACIÓN --------
ruta_archivo = "C:/Users/pepeb/OneDrive/Documentos/GitHub/NanoRob_Hadrones/WRO2025/LecturasPID.txt"

# -------- LECTURA DEL ARCHIVO --------
with open(ruta_archivo, "r") as archivo:
    lineas = [linea.strip() for linea in archivo if linea.strip()]

# Rango de búsqueda de parámetros PD

kp = 0

velocidad = float(lineas[-5])
Hz = float(lineas[-1])

valores_kp = [(velocidad**(5/6))/400, (velocidad**(5/6))/420, (velocidad**(5/6))/440, (velocidad**(5/6))/460,(velocidad**(5/6))/480]
valores_kd = [kp * 40, kp * 50, kp * 60, kp * 70, kp * 80, kp * 90]

errores = [float(l.split(";")[0]) if ";" in l else float(l) for l in lineas[:-5]]
tiempos = [i / Hz for i in range(len(errores))]

# -------- AUTO-TUNING PD --------
mejor_config = None
mejor_error = float("inf")

resultados = []

for kp, kd in itertools.product(valores_kp, valores_kd):
    salida = []
    error_anterior = errores[0]

    for i in range(len(errores)):
        error = errores[i]
        derivada = (error - error_anterior) * Hz  # derivada = delta_error / delta_t
        correccion = kp * error + kd * derivada
        salida.append(correccion)
        error_anterior = error

    rmse = mean_squared_error([0]*len(salida), salida, squared=False)
    resultados.append((kp, kd, rmse))

    if rmse < mejor_error:
        mejor_error = rmse
        mejor_config = (kp, kd)

# -------- RESULTADOS --------
print(">>> Mejores parámetros PD encontrados:")
print(f"kp = {mejor_config[0]}")
print(f"kd = {mejor_config[1]}")
print(f"Error (RMSE): {mejor_error:.4f}")

# -------- GRÁFICA DE LA MEJOR RESPUESTA --------
# Volver a calcular la respuesta con los mejores PID
kp, kd = mejor_config
salida = []
error_anterior = errores[0]

for i in range(len(errores)):
    error = errores[i]
    derivada = (error - error_anterior) * Hz
    correccion = kp * error + kd * derivada
    salida.append(correccion)
    error_anterior = error

plt.figure(figsize=(10, 5))
plt.plot(tiempos, salida, label="Corrección PD óptima", color="green")
plt.xlabel("Tiempo (s)")
plt.ylabel("Corrección")
plt.title(f"PD Autotune Result: kp={kp}, kd={kd}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pd_autotune_result.png", dpi=300)
plt.show()
