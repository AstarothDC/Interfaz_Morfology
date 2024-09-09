from PyQt6.QtWidgets import QMainWindow, QApplication,QLineEdit,QMessageBox,QTableWidget,QFileDialog,QTableWidgetItem

from PyQt6.QtGui import QGuiApplication,QIcon,QImage,QPixmap
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QPropertyAnimation, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
from PyQt6.uic import loadUi
#import recursos
import cv2

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
import os
from torchvision import transforms as torchtrans  
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from PIL import Image
from functions.hojas import model_hojas
from functions.tallos import modelo

np.set_printoptions(precision=8)

from ultralytics import YOLO

"""/// Listado de variable  utilizadas y funciones\\
    self.ruta_variable => Variable que guarda la dir de folder
    self.list_image    => Variable que contiene la lista de imagenes
    self.imgproyectada => variable para determinar que imagen se esta visualizando
    self.pag = dice en que pagina esta y reinicia
    timage= saber si es la imagen original o la procesada
    """

def annotated(results):

    annotated_frame = results['result_detection'][0].plot()
   
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 5
    color = (255, 255, 255)
    thickness = 25

    for result in results.get("list_direc"):

        org = (result.get("delt"),result.get("x_2"))
        annotated_frame = cv2.putText(annotated_frame, str(result.get("id")), org, font, fontScale, color, thickness, cv2.LINE_AA)


    return(annotated_frame)


class mainUI(QMainWindow):
    def __init__(self):
        super(mainUI,self).__init__()
        loadUi('Main_GUI.ui',self)
        self.ruta_img=None
        self.upload_img_steam.clicked.connect(self.leer_img)  # Boton para seleccionar la carpeta
        self.upload_img.clicked.connect(self.leer_img)  # Boton para seleccionar la carpeta

        self.hojas.clicked.connect(self.prediction_h)  # Boton para correr modelo de hojas
        self.tallos.clicked.connect(self.prediction_t)  # Boton para correr modelo de tallos

        self.b_home.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_home))
        self.b_home.clicked.connect(self.fun_home)
        self.b_model1.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_Hojas))
        self.b_model2.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_Tallos))
        self.b_information.clicked.connect(self.open_pdf)

        self.modeld1= YOLO("./models/Hojas_seg.pt")
        self.modeld2= YOLO("./models/detection.pt")
        self.modeld3= YOLO("./models/tallos.pt")

    def open_pdf(self):
        os.startfile("Manual de usuario.pdf")

    def leer_img(self):
        self.ruta_imagen, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if  self.ruta_imagen:
            self.pruebafun()  

    def pruebafun(self):
        try:
            # Verifica si se ha seleccionado una imagen
            if self.ruta_imagen:
                # Carga la imagen seleccionada
                self.image = cv2.imread(self.ruta_imagen)
                
                # Convierte la imagen de BGR a RGB
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

                # Redimensiona la imagen para adaptarla al cuadro de visualización
                self.image = cv2.resize(self.image, (520, 444), interpolation=cv2.INTER_LINEAR)

                # Convierte la imagen a formato QImage para mostrarla en QLabel
                qformat = QImage.Format.Format_RGB888
                img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

                # Mostrar la imagen en los QLabel correspondientes (l_image_3 y l_image_2)
                self.l_image_3.setPixmap(QPixmap.fromImage(img))
                self.l_image_2.setPixmap(QPixmap.fromImage(img))

            else:
                mensaje = "No image selected"
                QMessageBox.warning(self, "Warning", mensaje)

        except Exception as e:
            mensaje = f"Error loading image: {str(e)}"
            QMessageBox.critical(self, "Error", mensaje)

    
    def prediction_h(self):
        try:
            # Verifica si se ha seleccionado una imagen
            if self.ruta_imagen:
                # Realiza la predicción con el modelo
                results = model_hojas(self.ruta_imagen, self.modeld1, self.modeld2)

                annotated_frame = annotated(results)              

                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)  # Asegurarse que esté en RGB
                annotated_frame = cv2.resize(annotated_frame, (520, 444), interpolation=cv2.INTER_LINEAR)

                qformat = QImage.Format.Format_RGB888
                img = QImage(annotated_frame, annotated_frame.shape[1], annotated_frame.shape[0], annotated_frame.strides[0], qformat)

                # Muestra la imagen anotada en el QLabel (l_image_2)
                self.l_image_3.setPixmap(QPixmap.fromImage(img))

                
                plant_id = []
                count = []
                self.tabla_d2_2.verticalHeader().setVisible(False)
                
                for re in results.get("list_direc"):
                    plant_id.append(re.get("id"))
                    count.append(re.get("count_leafl"))                    

                if plant_id and count:

                    while self.tabla_d2_2.rowCount() < len(plant_id):
                        self.tabla_d2_2.insertRow(self.tabla_d2_2.rowCount())

                    for i, (plant, hojas) in enumerate(zip(plant_id, count)):

                        print(f"Asignando en fila {i}: Planta {plant}, Hojas {hojas}")
                            
                            # Asignar los valores a las dos columnas existentes
                        #self.tabla_d2_2.setItem(i, 0, QTableWidgetItem(str(plant)))  # Columna 'Planta'
                        plant_item = QTableWidgetItem(str(plant))  
                        plant_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.tabla_d2_2.setItem(i, 0, plant_item)    # Columna 'Alto'

                        #self.tabla_d2_2.setItem(i, 1, QTableWidgetItem(str(hojas)))  # Columna 'Hojas de Banano'
                        hojas_item = QTableWidgetItem(str(hojas))  
                        hojas_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.tabla_d2_2.setItem(i, 1, hojas_item)    # Columna 'Alto'

                    # Forzar la actualización de la tabla
                    self.tabla_d2_2.repaint()
                else:
                    print("No hay datos para mostrar en la tabla.")

            else:
                QMessageBox.warning(self, "Warning", "No image selected")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during prediction: {str(e)}")


    def fun_home(self):
        self.hact=0

    def prediction_t(self):
        try:
            # Verifica si se ha seleccionado una imagen
            if self.ruta_imagen:

                distance_text = self.distance.text()
                try:
                    distance_value = float(distance_text)
                except ValueError:
                    QMessageBox.warning(self, "Invalid input", "Please enter a valid number for the distance.")
                    return  # Sale de la función si el valor no es válido

                # Realiza la predicción con el modelo
                results = modelo(self.ruta_imagen, self.modeld3, distance_value)

                # Obtén el frame anotado de los resultados
                annotated_frame = results['results'][0].plot()
                print(results)

                # Convertir el resultado (annotated_frame) para mostrarlo en el QLabel
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)  # Asegurarse que esté en RGB
                annotated_frame = cv2.resize(annotated_frame, (520, 444), interpolation=cv2.INTER_LINEAR)

                qformat = QImage.Format.Format_RGB888
                img = QImage(annotated_frame, annotated_frame.shape[1], annotated_frame.shape[0], annotated_frame.strides[0], qformat)

                # Muestra la imagen anotada en el QLabel (l_image_2)
                self.l_image_2.setPixmap(QPixmap.fromImage(img))

                self.tabla_d2.verticalHeader().setVisible(False)
                alto = results.get('alto')
                ancho = results.get('ancho')

                if alto is not None and ancho is not None:
                    if self.tabla_d2.rowCount() == 0:
                        self.tabla_d2.insertRow(0)

                    # Asignar los valores a las dos columnas de la tabla
                    # Asignar el valor de 'alto' y centrarlo
                    alto_item = QTableWidgetItem(str(alto))  
                    alto_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.tabla_d2.setItem(0, 0, alto_item)    # Columna 'Alto'

                    # Asignar el valor de 'ancho' y centrarlo
                    ancho_item = QTableWidgetItem(str(ancho))
                    ancho_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)  
                    self.tabla_d2.setItem(0, 1, ancho_item)   # Columna 'Ancho'

                    # Forzar la actualización de la tabla
                    self.tabla_d2.repaint()
                else:
                    print("No hay datos para mostrar en la tabla.")



            else:
                QMessageBox.warning(self, "Warning", "No image selected")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during prediction: {str(e)}")

    
    
    





        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = mainUI()
    ui.show()
    app.exec()