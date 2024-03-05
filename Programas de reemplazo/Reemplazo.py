import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.font as tkFont

def reemplazar_en_archivo(archivo, palabra_buscar, palabra_reemplazar, reemplazar_todas=False):
    try:
        with open(archivo, 'r', encoding='utf-8') as archivo_entrada:
            contenido = archivo_entrada.read()

        if reemplazar_todas:
            contenido_modificado = contenido.replace(palabra_buscar, palabra_reemplazar)
        else:
            contenido_modificado = contenido.replace(palabra_buscar, palabra_reemplazar)

        with open(archivo, 'w', encoding='utf-8') as archivo_salida:
            archivo_salida.write(contenido_modificado)

        if reemplazar_todas:
            messagebox.showinfo("Reemplazo exitoso", f'Se han reemplazado todas las ocurrencias de "{palabra_buscar}" por "{palabra_reemplazar}" en el archivo "{archivo}".')
        else:
            messagebox.showinfo("Reemplazo exitoso", f'Se ha reemplazado la primera ocurrencia de "{palabra_buscar}" por "{palabra_reemplazar}" en el archivo "{archivo}".')

    except FileNotFoundError:
        messagebox.showerror("Error", f'El archivo "{archivo}" no se encontró.')
    except Exception as e:
        messagebox.showerror("Error", f'Ocurrió un error: {str(e)}')

def seleccionar_archivo():
    archivo = filedialog.askopenfilename(title="Seleccionar archivo")
    if archivo:
        entry_archivo.delete(0, tk.END)
        entry_archivo.insert(0, archivo)

def ejecutar_reemplazo():
    archivo = entry_archivo.get()
    palabra_buscar = entry_palabra_buscar.get()
    palabra_reemplazar = entry_palabra_reemplazar.get()
    reemplazar_todas = check_var.get()

    reemplazar_en_archivo(archivo, palabra_buscar, palabra_reemplazar, reemplazar_todas)

app = tk.Tk()
app.title("Reemplazar en Archivo")

# Obtén las dimensiones de la pantalla
ancho_pantalla = app.winfo_screenwidth()
alto_pantalla = app.winfo_screenheight()

# Establece el tamaño de la ventana al máximo posible
app.geometry(f"{ancho_pantalla}x{alto_pantalla}")

frame_archivo = tk.Frame(app, pady=10)
frame_archivo.pack()

label_archivo = tk.Label(frame_archivo, text="Archivo:")
label_archivo.grid(row=0, column=0, padx=5, pady=5)

entry_archivo = tk.Entry(frame_archivo, width=30)
entry_archivo.grid(row=0, column=1, padx=5, pady=5)

button_seleccionar = tk.Button(frame_archivo, text="Seleccionar archivo", command=seleccionar_archivo)
button_seleccionar.grid(row=0, column=2, padx=5, pady=5)

frame_palabras = tk.Frame(app, pady=10)
frame_palabras.pack()

label_palabra_buscar = tk.Label(frame_palabras, text="Palabra a buscar:")
label_palabra_buscar.grid(row=0, column=0, padx=5, pady=5)

entry_palabra_buscar = tk.Entry(frame_palabras, width=30)
entry_palabra_buscar.grid(row=0, column=1, padx=5, pady=5)

label_palabra_reemplazar = tk.Label(frame_palabras, text="Palabra a reemplazar:")
label_palabra_reemplazar.grid(row=1, column=0, padx=5, pady=5)

entry_palabra_reemplazar = tk.Entry(frame_palabras, width=30)
entry_palabra_reemplazar.grid(row=1, column=1, padx=5, pady=5)

frame_opciones = tk.Frame(app, pady=10)
frame_opciones.pack()

check_var = tk.BooleanVar()
check_var.set(False)

check_reemplazar_todas = tk.Checkbutton(frame_opciones, text="Reemplazar todas las coincidencias", variable=check_var)
check_reemplazar_todas.pack()

button_ejecutar = tk.Button(app, text="Ejecutar Reemplazo", command=ejecutar_reemplazo)
button_ejecutar.pack()

app.mainloop()
