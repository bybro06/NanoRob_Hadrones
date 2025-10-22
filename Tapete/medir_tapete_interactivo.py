#!/usr/bin/env python3
"""
medir_tapete_pyqt.py

PyQt6 application "Medidor Tapete Pro".

Características:
 - Carga imagen (PNG/JPG)
 - Zoom y pan
 - Clic izquierdo: añadir puntos (P1, P2, ...)
 - Clic derecho: eliminar último punto
 - Botones: Abrir, Guardar captura, Exportar CSV, Reiniciar, Deshacer
 - Muestra distancias entre puntos consecutivos (cm) y ángulos interiores (usar 'd' como símbolo)
 - Escala por las dimensiones reales del tapete (por defecto 2362mm x 1143mm), configurables
 - Visuales nítidos usando QPainter (fondo redondeado, antialias)

Dependencias:
    pip install PyQt6 opencv-python numpy pillow

Ejecutar:
    python medir_tapete_pyqt.py

"""

import sys
import os
import math
import csv
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont, QAction, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMenuBar, QMenu, QStatusBar,
    QGraphicsView, QGraphicsScene, QDoubleSpinBox, QSpinBox, QToolBar, QMessageBox
)

# --- Defaults
DEFAULT_WIDTH_MM = 2362.0
DEFAULT_HEIGHT_MM = 1143.0

# --- Utility functions
def cv2_to_qimage(cv_img: np.ndarray) -> QImage:
    """Convert BGR cv2 image to QImage"""
    if cv_img.ndim == 2:
        h, w = cv_img.shape
        bytes_per_line = w
        return QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8).copy()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cv_img = None
        self.pixmap = None
        self.points: List[Tuple[int,int]] = []
        self._zoom = 1.0
        self._offset = QPoint(0,0)
        self._drag_last = None

        self.real_w_mm = DEFAULT_WIDTH_MM
        self.real_h_mm = DEFAULT_HEIGHT_MM

        self.setMouseTracking(True)

    def load_image(self, path: str):
        # abrir imagen con PIL para evitar problemas con unicode/Windows
        try:
            pil_img = Image.open(path).convert('RGB')
            cv_img = np.array(pil_img)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir imagen: {e}")

        self.cv_img = cv_img
        self.points = []
        self._zoom = 1.0
        self._offset = QPoint(0,0)
        self.adjust_zoom_to_fit()
        self.update()

    def adjust_zoom_to_fit(self):
        """Ajusta zoom para que la imagen ocupe el widget"""
        if self.cv_img is None:
            return
        img_h, img_w = self.cv_img.shape[:2]
        widget_w = self.width() if self.width()>0 else img_w
        widget_h = self.height() if self.height()>0 else img_h
        scale_w = widget_w / img_w
        scale_h = widget_h / img_h
        self._zoom = min(scale_w, scale_h)

    def set_real_dimensions(self, w_mm: float, h_mm: float):
        self.real_w_mm = float(w_mm)
        self.real_h_mm = float(h_mm)
        self.update()

    def reset(self):
        self.points = []
        self._zoom = 1.0
        self._offset = QPoint(0,0)
        self.adjust_zoom_to_fit()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        if self.cv_img is None:
            return

        # pixmap original
        self.pixmap = QPixmap.fromImage(cv2_to_qimage(self.cv_img))
        pw = int(self.pixmap.width()*self._zoom)
        ph = int(self.pixmap.height()*self._zoom)
        scaled = self.pixmap.scaled(pw, ph, Qt.AspectRatioMode.KeepAspectRatio)

        # centrar imagen + offset
        x = (self.width() - scaled.width())//2 + self._offset.x()
        y = (self.height() - scaled.height())//2 + self._offset.y()
        painter.drawPixmap(x, y, scaled)

        # escala mm/pixel
        img_h, img_w = self.cv_img.shape[:2]
        mm_per_px_x = self.real_w_mm / img_w
        mm_per_px_y = self.real_h_mm / img_h

        # dibujar líneas, puntos y distancias
        for i, p in enumerate(self.points):
            px, py = p
            sx = x + int(px * self._zoom)
            sy = y + int(py * self._zoom)
            painter.setBrush(QBrush(QColor(0,0,200)))
            painter.setPen(QPen(QColor(0,0,0),1))
            painter.drawEllipse(QPoint(sx,sy), max(3,int(3*self._zoom)), max(3,int(3*self._zoom)))
            painter.setPen(QPen(QColor(0,0,200),1))
            painter.setFont(QFont('Sans', max(8,int(10*self._zoom))))
            painter.drawText(sx+6, sy-6, f"P{i+1}")

        for i in range(1, len(self.points)):
            p1 = self.points[i-1]; p2 = self.points[i]
            sx1 = x + int(p1[0]*self._zoom); sy1 = y + int(p1[1]*self._zoom)
            sx2 = x + int(p2[0]*self._zoom); sy2 = y + int(p2[1]*self._zoom)
            # línea
            painter.setPen(QPen(QColor(255,120,120), max(1,int(2*self._zoom))))
            painter.drawLine(sx1, sy1, sx2, sy2)

            # distancia en cm
            dx_px = p2[0]-p1[0]; dy_px = p2[1]-p1[1]
            dx_mm = dx_px * mm_per_px_x; dy_mm = -dy_px * mm_per_px_y
            dist_cm = math.hypot(dx_mm, dy_mm)/10.0
            dist_text = f"{dist_cm:.1f} cm"

            # midpoint con offset perpendicular
            mx = (sx1+sx2)//2; my = (sy1+sy2)//2
            seg_len = math.hypot(sx2-sx1, sy2-sy1)
            if seg_len==0: seg_len=1
            nx = -(sy2-sy1)/seg_len; ny = (sx2-sx1)/seg_len
            off = int(15*self._zoom)
            tx = int(mx+nx*off)
            ty = int(my+ny*off)
            self._draw_text_box(painter, dist_text, tx, ty, bg_color=(30,30,30,180), fg=(255,255,255))

        # ángulos
        for i in range(1,len(self.points)-1):
            p_prev = np.array(self.points[i-1],dtype=float)
            p_curr = np.array(self.points[i],dtype=float)
            p_next = np.array(self.points[i+1],dtype=float)
            v1 = p_prev - p_curr; v2 = p_next - p_curr
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1==0 or n2==0: continue
            v1n = v1/n1; v2n = v2/n2
            ang_between = math.degrees(math.acos(np.clip(np.dot(v1n,v2n),-1.0,1.0)))
            interior_ang = 180.0 - ang_between
            angle_text = f"{interior_ang:.1f}d"
            bis = v1n+v2n
            if np.linalg.norm(bis)==0: continue
            bisn = bis/np.linalg.norm(bis)
            bx = int(p_curr[0]+bisn[0]*60/self._zoom)
            by = int(p_curr[1]+bisn[1]*60/self._zoom)
            sx_b = x + int(bx*self._zoom); sy_b = y + int(by*self._zoom)
            self._draw_text_box(painter, angle_text, sx_b, sy_b, bg_color=(0,100,0,200), fg=(255,255,255))

        painter.end()

    def _draw_text_box(self, painter: QPainter, text: str, cx: int, cy: int, bg_color=(0,0,0,180), fg=(255,255,255)):
        font = QFont('Sans', max(8,int(10*self._zoom)))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(text)
        th = metrics.height()
        pad = max(4,int(4*self._zoom))
        x1 = cx - tw//2 - pad; y1 = cy - th//2 - pad
        x2 = cx + tw//2 + pad; y2 = cy + th//2 + pad
        rect = QRect(x1,y1,x2-x1,y2-y1)
        painter.setBrush(QBrush(QColor(*bg_color)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, max(4,int(6*self._zoom)), max(4,int(6*self._zoom)))
        painter.setPen(QPen(QColor(0,0,0),2))
        painter.drawText(cx - tw//2, cy + th//4, text)
        painter.setPen(QPen(QColor(*fg),1))
        painter.drawText(cx - tw//2, cy + th//4, text)

    # --- Mouse events
    def mousePressEvent(self, ev):
        if self.cv_img is None: return
        if ev.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(ev.position())
            if img_pt: self.points.append((int(img_pt[0]), int(img_pt[1])))
            self.update()
        elif ev.button() == Qt.MouseButton.RightButton:
            if self.points: self.points.pop()
            self.update()
        elif ev.button() == Qt.MouseButton.MiddleButton:
            self._drag_last = ev.position()

    def mouseMoveEvent(self, ev):
        if self._drag_last and ev.buttons() & Qt.MouseButton.MiddleButton:
            delta = ev.position() - self._drag_last
            self._offset += QPoint(int(delta.x()), int(delta.y()))
            self._drag_last = ev.position()
            self.update()

    def mouseReleaseEvent(self, ev):
        self._drag_last = None

    def wheelEvent(self, ev):
        angle = ev.angleDelta().y()
        factor = 1.0 + (angle/240.0)
        old_zoom = self._zoom
        self._zoom = max(0.1, min(5.0, self._zoom*factor))
        pos = ev.position()
        img_before = self._widget_to_image(pos)
        self.update()
        img_after = self._widget_to_image(pos)
        if img_before and img_after:
            dx = img_after[0]-img_before[0]; dy = img_after[1]-img_before[1]
            self._offset += QPoint(int(dx*self._zoom), int(dy*self._zoom))
        self.update()

    def export_overlay_image(self, path: str):
        if self.pixmap is None:
            return
        # crear un QPixmap del tamaño del widget
        pix = QPixmap(self.size())
        self.render(pix)  # renderiza el widget completo
        pix.save(path)

    def export_csv(self, path: str):
        """Exporta los puntos y distancias a CSV"""
        if self.cv_img is None or len(self.points) < 2:
            return

        h, w = self.cv_img.shape[:2]
        mm_per_px_x = self.real_w_mm / w
        mm_per_px_y = self.real_h_mm / h

        rows = []
        for i in range(1, len(self.points)):
            p1 = self.points[i-1]
            p2 = self.points[i]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dx_mm = dx * mm_per_px_x
            dy_mm = -dy * mm_per_px_y
            dist_cm = math.hypot(dx_mm, dy_mm) / 10.0
            rows.append([i, p1[0], p1[1], p2[0], p2[1], f"{dist_cm:.3f}"])

        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['seg', 'x1', 'y1', 'x2', 'y2', 'dist_cm'])
            writer.writerows(rows)

    def _widget_to_image(self, qp) -> Tuple[float,float]:
        if self.pixmap is None: return None
        x = (self.width()-self.pixmap.width()*self._zoom)//2 + self._offset.x()
        y = (self.height()-self.pixmap.height()*self._zoom)//2 + self._offset.y()
        imgx = (qp.x() - x)/self._zoom
        imgy = (qp.y() - y)/self._zoom
        img_h, img_w = self.cv_img.shape[:2]
        if imgx<0 or imgy<0 or imgx>=img_w or imgy>=img_h: return None
        return (imgx,imgy)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medidor Tapete Pro")
        self.resize(1200,800)

        self.image_widget = ImageWidget()

        toolbar = QToolBar()
        self.addToolBar(toolbar)

        open_action = QAction("Abrir", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        save_action = QAction("Guardar captura", self)
        save_action.triggered.connect(self.save_capture)
        toolbar.addAction(save_action)

        csv_action = QAction("Exportar CSV", self)
        csv_action.triggered.connect(self.export_csv)
        toolbar.addAction(csv_action)

        reset_action = QAction("Reiniciar", self)
        reset_action.triggered.connect(self.image_widget.reset)
        toolbar.addAction(reset_action)

        undo_action = QAction("Deshacer", self)
        undo_action.triggered.connect(self.undo_point)
        toolbar.addAction(undo_action)

        # dimensiones reales
        wspin = QDoubleSpinBox()
        wspin.setRange(10,10000); wspin.setValue(DEFAULT_WIDTH_MM); wspin.setSuffix(" mm")
        wspin.valueChanged.connect(lambda v: self.image_widget.set_real_dimensions(v,self.image_widget.real_h_mm))
        toolbar.addWidget(wspin)
        hspin = QDoubleSpinBox()
        hspin.setRange(10,10000); hspin.setValue(DEFAULT_HEIGHT_MM); hspin.setSuffix(" mm")
        hspin.valueChanged.connect(lambda v: self.image_widget.set_real_dimensions(self.image_widget.real_w_mm,v))
        toolbar.addWidget(hspin)

        central = QWidget()
        vlay = QVBoxLayout()
        vlay.addWidget(self.image_widget)
        central.setLayout(vlay)
        self.setCentralWidget(central)

    def open_image(self):
        path,_ = QFileDialog.getOpenFileName(self,"Abrir imagen","","Imagenes (*.png *.jpg *.jpeg *.bmp)")
        if not path: return
        try:
            self.image_widget.load_image(path)
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

    def save_capture(self):
        path, _ = QFileDialog.getSaveFileName(self, "Guardar captura", "", "PNG (*.png);;JPG (*.jpg)")
        if not path:
            return
        # llama a la función del widget para guardar la imagen con overlay
        self.image_widget.export_overlay_image(path)
        QMessageBox.information(self, "Guardado", f"Captura guardada en:\n{path}")

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Exportar CSV", "", "CSV (*.csv)")
        if not path:
            return
        # llama a la función del widget para exportar CSV
        self.image_widget.export_csv(path)
        QMessageBox.information(self, "Exportado", f"CSV guardado en:\n{path}")


    def undo_point(self):
        if self.image_widget.points:
            self.image_widget.points.pop()
            self.image_widget.update()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()