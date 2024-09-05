
from ultralytics import YOLO
import math

def calcular_distancia(objeto1, objeto2):
    # Calcula la distancia entre los centros de dos objetos
    x1, y1 = objeto1[0], objeto1[1]  # coordenadas del centro del objeto1
    x2, y2 = objeto2[0], objeto2[1]  # coordenadas del centro del objeto2
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distancia

def modelo(imagen_path, model_path, distancia_minima=40):


    # prediccion del modelo
    results = model_path(imagen_path,iou=0.3)  # predict on an image

    # Inicializar la lista para almacenar los resultados
    resultados_objetos = []

    for obj_idx, mascaras in enumerate(results[0].masks):
        sub_x, sub_y = zip(*mascaras.xy[0])

        sec_1_min = min(sub_y)
        sec_1_max = sec_1_min + (max(sub_y) - sec_1_min) / 3
        sec_2_min = sec_1_max
        sec_2_max = sec_2_min + (max(sub_y) - sec_1_min) / 3
        sec_3_min = sec_2_max
        sec_3_max = max(sub_y)

        sec1_indices_x = [i for i, y in enumerate(sub_y) if sec_1_min <= y < sec_1_max]
        sec2_indices_x = [i for i, y in enumerate(sub_y) if sec_2_min <= y < sec_2_max]
        sec3_indices_x = [i for i, y in enumerate(sub_y) if sec_3_min <= y <= sec_3_max]

        def calcular_ancho_sec(indices_x):
            return max(sub_x[i] for i in indices_x) - min(sub_x[i] for i in indices_x)

        ancho_sec1 = calcular_ancho_sec(sec1_indices_x)
        ancho_sec2 = calcular_ancho_sec(sec2_indices_x)
        ancho_sec3 = calcular_ancho_sec(sec3_indices_x)

        prom_sec_objeto = (ancho_sec1 + ancho_sec2 + ancho_sec3) / 3
        altura_objeto = max(sub_y) - min(sub_y)

        #ecuaciones de calibracion en base a la distancia        
        alt = (0.004532728*distancia_minima)-0.021187199
        anch = (0.004534844*distancia_minima)-0.029351267
        # Calibrar ancho y altura
        ancho_calibrado = prom_sec_objeto * anch
        altura_calibrada = altura_objeto * alt

        descartar_objeto = False
        objetos_cercanos = []

        for otro_objeto in resultados_objetos:
            distancia_entre_objetos = calcular_distancia((ancho_calibrado, altura_calibrada), otro_objeto)
            if distancia_entre_objetos < distancia_minima:
                # Si está demasiado cerca, agregar a la lista de objetos cercanos
                objetos_cercanos.append(otro_objeto)

        if objetos_cercanos:
            # Si hay objetos cercanos, mantener solo el más cercano y descartar los demás
            objeto_cercano_mas_cercano = min(objetos_cercanos, key=lambda x: calcular_distancia((ancho_calibrado, altura_calibrada), x))
            descartar_objeto = True

        if not descartar_objeto:
            # Si no hay objetos cercanos, añadir el objeto a la lista
            resultados_objetos.append((ancho_calibrado, altura_calibrada))
            # Mensajes de depuración para cada objeto


    response = {'results':results, 
                'alto':altura_calibrada,
                 'ancho':ancho_calibrado }
    
    return response