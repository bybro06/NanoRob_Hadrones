#!/usr/bin/env python3
"""
Medidor interactivo de distancias y ángulos sobre una imagen de tapete.
- Muestra distancias entre puntos consecutivos (cm).
- Muestra ángulos interiores (180° - ángulo medido) entre rectas consecutivas.
- Fondo redondeado y texto nítido.
"""

import cv2
import numpy as np
import math
import argparse
import os
import tkinter as tk

# --- Parámetros reales (mm)
DEFAULT_REAL_WIDTH_MM = 2362.0
DEFAULT_REAL_HEIGHT_MM = 1143.0

def mm_to_cm(mm): 
    return mm / 10.0

def angle_between_vectors_deg(v1, v2):
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cosang = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def draw_rounded_box(img, x1, y1, x2, y2, color=(0,0,0), radius=6, alpha=0.5):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_text_with_bg(img, text, pos, font, scale, color_text, thickness=1,
                      bg_color=(0, 0, 0), alpha=0.5, pad=4, radius=6):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    x1 = x - pad
    y1 = y - th - pad
    x2 = x + tw + pad
    y2 = y + pad
    draw_rounded_box(img, x1, y1, x2, y2, bg_color, radius, alpha)
    cv2.putText(img, text, (x, y), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color_text, thickness, cv2.LINE_AA)

class Medidor:
    def __init__(self, img, real_width_mm=DEFAULT_REAL_WIDTH_MM, real_height_mm=DEFAULT_REAL_HEIGHT_MM):
        self.orig = img.copy()
        self.img = img
        self.h, self.w = img.shape[:2]
        self.real_w_mm = float(real_width_mm)
        self.real_h_mm = float(real_height_mm)
        self.scale_x = self.real_w_mm / self.w
        self.scale_y = self.real_h_mm / self.h
        self.points = []

    def reset(self):
        self.points = []
        self.img = self.orig.copy()

    def add_point(self, x, y):
        self.points.append((int(x), int(y)))
        self.redraw()

    def pixel_to_real_vector_mm(self, p_from, p_to):
        dx_px = p_to[0] - p_from[0]
        dy_px = p_to[1] - p_from[1]
        dx_mm = dx_px * self.scale_x
        dy_mm = -dy_px * self.scale_y
        return np.array([dx_mm, dy_mm])

    def pixel_distance_mm(self, p1, p2):
        v = self.pixel_to_real_vector_mm(p1, p2)
        return float(np.linalg.norm(v))

    def redraw(self):
        self.img = self.orig.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1

        # Dibujar puntos
        for i, p in enumerate(self.points):
            cv2.circle(self.img, p, 5, (0, 0, 230), -1)
            cv2.putText(self.img, f"P{i+1}", (p[0]+6, p[1]-6),
                        font, 0.5, (0,0,230), 1, cv2.LINE_AA)

        # Dibujar líneas y distancias
        for i in range(1, len(self.points)):
            p1, p2 = self.points[i-1], self.points[i]
            cv2.line(self.img, p1, p2, (255,130,130), 2)

            dist_mm = self.pixel_distance_mm(p1, p2)
            dist_cm = mm_to_cm(dist_mm)
            text = f"{dist_cm:.1f} cm"

            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)

            # Mover texto un poco perpendicular a la línea
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length > 0:
                dx, dy = dx / length, dy / length
                perp_x, perp_y = -dy, dx
                mid_x += int(perp_x * 20)
                mid_y += int(perp_y * 20)

            draw_text_with_bg(self.img, text, (mid_x, mid_y),
                              font, scale, (255,255,255), thickness,
                              bg_color=(50,50,50), alpha=0.6)

        # Dibujar ángulos
        for i in range(1, len(self.points) - 1):
            p_prev = np.array(self.points[i-1], dtype=float)
            p_curr = np.array(self.points[i], dtype=float)
            p_next = np.array(self.points[i+1], dtype=float)

            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                continue

            v1n = v1 / n1
            v2n = v2 / n2
            ang_between = math.degrees(math.acos(np.clip(np.dot(v1n, v2n), -1.0, 1.0)))
            interior_ang = 180.0 - ang_between

            bisector = v1n + v2n
            if np.linalg.norm(bisector) == 0:
                continue
            bisector /= np.linalg.norm(bisector)

            offset = 60
            text_center = (int(p_curr[0] + bisector[0]*offset),
                           int(p_curr[1] + bisector[1]*offset))

            angle_text = f"{interior_ang:.1f}d"
            draw_text_with_bg(self.img, angle_text,
                              (text_center[0], text_center[1]),
                              font, scale, (255,255,255), thickness,
                              bg_color=(0,80,0), alpha=0.6)

def mouse_callback(event, x, y, flags, param):
    med: Medidor = param
    if event == cv2.EVENT_LBUTTONDOWN:
        med.add_point(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        med.reset()

def main():
    parser = argparse.ArgumentParser(description='Medidor interactivo de tapete con ángulos y distancias')
    parser.add_argument('image', help='ruta de la imagen PNG/JPG')
    parser.add_argument('--width_mm', type=float, default=DEFAULT_REAL_WIDTH_MM)
    parser.add_argument('--height_mm', type=float, default=DEFAULT_REAL_HEIGHT_MM)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print('ERROR: imagen no encontrada:', args.image)
        return

    img = cv2.imread(args.image)
    if img is None:
        print('ERROR: no se pudo abrir la imagen.')
        return

    # Ajustar al largo de pantalla
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    h, w = img.shape[:2]
    scale = min(screen_w / w, screen_h / h, 1.0)
    if scale < 1.0:
        img_display = cv2.resize(img, (int(w*scale), int(h*scale)))
    else:
        img_display = img.copy()

    med = Medidor(img_display, args.width_mm, args.height_mm)

    winname = 'Medidor Tapete (clic izq: punto, der/r: reiniciar, q/ESC: salir)'
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(winname, mouse_callback, med)

    while True:
        cv2.imshow(winname, med.img)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            med.reset()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
