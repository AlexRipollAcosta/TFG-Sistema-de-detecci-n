# Sistema de Detección en Tiempo Real con Múltiples Modelos de IA
# Combina YOLO (detección de objetos), FastFlowNet (flujo óptico),
# ByteTracker (seguimiento) y Depth Anything V2 (estimación de profundidad)

# ======================== IMPORTACIONES ========================
import argparse                     # Para argumentos de línea de comandos
import cv2                         # OpenCV para procesamiento de imágenes
import numpy as np                 # Numpy para operaciones matemáticas
import torch                       # PyTorch para deep learning
import time                        # Para medición de tiempos
from types import SimpleNamespace   # Para crear objetos con atributos dinámicos
from ultralytics import YOLO        # YOLO de Ultralytics
from ultralytics.trackers.byte_tracker import BYTETracker  # Tracker de objetos
from PIL import Image, ImageTk      # Para manipulación de imágenes en GUI
import tkinter as tk               # GUI principal
from fastflownet import FastFlowNet # Modelo para flujo óptico
import torchvision.transforms as transforms
import tensorrt as trt             # TensorRT para optimización de modelos
import pycuda.driver as cuda       # CUDA para operaciones GPU
import pycuda.autoinit             # Inicialización automática de CUDA
from collections import OrderedDict, namedtuple
import os
import matplotlib                  # Para mapas de colores
from utils.torch_utils import select_device

# ======================== CONFIGURACIÓN GLOBAL ========================
# Rutas de los modelos y archivos
TRT_ENGINE_PATH_YOLO = "/imagenes/yolov7/yolov7-tiny-nms.trt"
FASTFLOWNET_PATH = "/imagenes/FastFlowNet/checkpoints/fastflownet_ft_mix.pth"
TRT_ENGINE_PATH_DEPTH_ANYTHING_V2 = "/imagenes/Depth-Anything-V2/metric_depth/model.trt"
VIDEO_PATH = "/imagenes/dataset1/test/videos/eeee.mp4"

# Configuración de tamaños y parámetros
TARGET_SIZE_640 = (640, 640)
TARGET_SIZE_518 = (518, 518)
RELATION_640_518 = 640/518
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
TRACKER_CONF_THRESHOLD = 0.4
MAX_MOVEMENT_RATIO = 0.2
FRAME_AGE = 30
cmap = matplotlib.colormaps.get_cmap('Spectral_r')


# ======================== INTERFAZ GRÁFICA TKINTER ========================
class DetectionApp:
    """
    Clase para manejar la interfaz gráfica de usuario con Tkinter.
    Muestra detecciones, flujo óptico, profundidad e información de tracking.
    """
    
    def __init__(self, root, w_raw_image, h_raw_image):
        """
        Inicializa la aplicación GUI con paneles para diferentes visualizaciones.
        
        Args:
            root: Ventana principal de Tkinter
            w_raw_image: Ancho de la imagen original
            h_raw_image: Alto de la imagen original
        """
        self.root = root
        self.root.title("Sistema de Detección en Tiempo Real")
        
        # Configurar tamaño de ventana (2x ancho para mostrar múltiples paneles)
        app_width = 2 * w_raw_image
        app_height = h_raw_image
        self.root.geometry(f"{app_width}x{app_height}")

        # Panel de información izquierda (muestra estadísticas de tracking)
        info_width_px = w_raw_image / 2
        info_height_px = h_raw_image

        self.info_frame = tk.Frame(root, width=info_width_px, height=info_height_px, bg='black')
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.info_frame.pack_propagate(False)  # Mantener tamaño fijo

        # Área de texto para mostrar información con estilo consola
        self.info_panel = tk.Text(self.info_frame, bg='black', fg='lime', font=("Consolas", 10), bd=0)
        self.info_panel.pack(fill=tk.BOTH, expand=True)

        # Canvas para mostrar imagen con detecciones
        self.detection_canvas = tk.Label(root)
        self.detection_canvas.pack(side=tk.LEFT)

        # Panel derecho para movimiento y profundidad
        self.right_panel = tk.Frame(root)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH)

        # Canvas para visualización de flujo óptico
        self.motion_canvas = tk.Label(self.right_panel)
        self.motion_canvas.pack()

        # Canvas para visualización de profundidad
        self.depth_canvas = tk.Label(self.right_panel)
        self.depth_canvas.pack()

    def update_detection_image(self, frame_bgr):
        """Actualiza la imagen de detecciones en el canvas principal."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB para Tkinter
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.detection_canvas.configure(image=img)
        self.detection_canvas.image = img  # Mantener referencia para evitar garbage collection

    def update_motion_image(self, flow_vis_rgb):
        """Actualiza la visualización del flujo óptico."""
        img = ImageTk.PhotoImage(Image.fromarray(flow_vis_rgb))
        self.motion_canvas.configure(image=img)
        self.motion_canvas.image = img

    def update_depth_image(self, depth_vis_rgb):
        """Actualiza la visualización del mapa de profundidad."""
        img = ImageTk.PhotoImage(Image.fromarray(depth_vis_rgb))
        self.depth_canvas.configure(image=img)
        self.depth_canvas.image = img

    def update_info_panel(self, track_data):
        """
        Actualiza el panel de información con estadísticas de tracking.
        
        Args:
            track_data: Lista de tuplas (id, clase, confianza, movimiento, distancia)
        """
        from collections import Counter

        self.info_panel.delete(1.0, tk.END)  # Limpiar contenido anterior

        # Contar total de unidades detectadas
        total_unidades = len(track_data)
        self.info_panel.insert(tk.END, f"UNIDADES DETECTADAS: {total_unidades}\n")

        # Contar unidades por clase
        clase_counter = Counter(cls for _, cls, *_ in track_data)

        # Mostrar resumen estadístico por clase
        self.info_panel.insert(tk.END, "=" * 30 + "\n", "divider")
        self.info_panel.insert(tk.END, "Resumen por clase:\n", "section")
        self.info_panel.insert(tk.END, f"Infantry: {clase_counter.get('Infantry', 0)}\n")
        self.info_panel.insert(tk.END, f"Tank: {clase_counter.get('Tank', 0)}\n")
        self.info_panel.insert(tk.END, f"Mechanized Inf: {clase_counter.get('Mechanized Inf', 0)}\n")
        self.info_panel.insert(tk.END, "=" * 30 + "\n\n", "divider")

        # Mostrar detalles individuales de cada objeto trackeado
        for tid, cls, conf, mov, dist in track_data:
            info = (
                f"ID: {tid}\n"
                f"Clase: {cls}\n"
                f"Confianza: {conf:.2f}%\n"
                f"Distancia: {int(round(dist))}m\n"
                f"Mov: {int(round(mov))}%\n\n"
            )
            self.info_panel.insert(tk.END, info)

        # Configurar estilos de texto
        self.info_panel.tag_config("divider", foreground="green")
        self.info_panel.tag_config("section", font=("Helvetica", 10, "bold"))

# ======================== GESTOR DE RECURSOS GPU ========================
class GPUResourceManager:
    """
    Clase para gestionar recursos compartidos de GPU entre múltiples modelos TensorRT.
    Evita la duplicación de recursos y optimiza el uso de memoria GPU.
    """

    def __init__(self):
        # Logger compartido para todos los motores TensorRT
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        # Runtime compartido para cargar engines
        self.runtime = trt.Runtime(self.logger)
        # Stream CUDA global para operaciones asíncronas
        self.stream = cuda.Stream()

    def load_trt_engine(self, engine_path):
        """
        Carga un motor TensorRT desde archivo.
        
        Args:
            engine_path: Ruta al archivo .trt
            
        Returns:
            Motor TensorRT deserializado
        """
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"El archivo del engine no se encuentra en {engine_path}")
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
            engine = self.runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                raise RuntimeError("Error al deserializar el engine")
            return engine

# ======================== WRAPPER PARA BYTETRACKER ========================
class TrackerWrapper:
    """
    Envoltorio para el tracker ByteTracker que maneja el seguimiento de objetos
    entre frames, asociando detecciones con IDs únicos.
    """

    def __init__(self, frame_rate=20):
        """
        Inicializa el tracker con parámetros optimizados.
        
        Args:
            frame_rate: FPS del video para calibrar el tracker
        """
        # Configuración del tracker ByteTracker
        self.args = SimpleNamespace(
            tracker_type='bytetrack',
            track_high_thresh=0.5,     # Umbral alto para tracking confiable
            track_low_thresh=0.4,       # Umbral bajo para recuperar tracks perdidos
            new_track_thresh=0.6,      # Umbral para crear nuevos tracks
            track_buffer=FRAME_AGE,     # Frames que se mantiene un track sin detección
            match_thresh=0.8,           # Umbral de matching entre detecciones y tracks
            fuse_score=True             # Fusionar scores de detección y tracking
        )
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)

    class Detections:
        """Clase auxiliar para encapsular datos de detección en formato esperado por ByteTracker."""
        def __init__(self, boxes, confidences, class_ids, movements):
            self.conf = confidences      # Confianzas de detección
            self.xywh = boxes           # Cajas en formato centro_x, centro_y, ancho, alto
            self.cls = class_ids        # IDs de clase
            self.movs = movements       # Información de movimiento

    def track(self, detection_data, frame):
        """
        Realiza el tracking usando ByteTracker.
        
        Args:
            detection_data: Objeto Detections con información de detecciones
            frame: Frame actual para contexto temporal
            
        Returns:
            Lista de tracks actualizados
        """
        detections = self.Detections(
            detection_data.xywh,
            detection_data.conf,
            detection_data.cls,
            detection_data.movs
        )
        return self.tracker.update(detections, frame)

    def get_iou(self, boxA, boxB):
        """
        Calcula Intersection over Union (IoU) entre dos cajas.
        Métrica fundamental para asociar detecciones con tracks existentes.
        """
        # Calcular coordenadas de intersección
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Calcular área de intersección
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        # Calcular áreas de cada caja
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Calcular IoU
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def associate_movements(self, tracks, original_detections):
        """
        Asocia información de movimiento de las detecciones originales con los tracks.
        Esto es necesario porque ByteTracker no preserva datos personalizados.
        """
        associated = []
        for track in tracks:
            # Asegurar formato correcto del track
            if len(track) >= 8:
                track = track[:7]  # Eliminar movement anterior si existe

            x1, y1, x2, y2, track_id, conf, clas_id = track
            track_box = [x1, y1, x2, y2]

            # Encontrar la detección original con mayor IoU
            best_iou = 0
            best_movement = None
            for det, movement in zip(original_detections.xywh, original_detections.movs):
                cx, cy, w, h = det
                det_box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                iou = self.get_iou(track_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_movement = movement

            # Agregar información de movimiento al track
            associated.append((x1, y1, x2, y2, track_id, conf, clas_id, best_movement))
        return associated

    def format_detections_for_tracker(self, detections, conf_threshold):
        """
        Formatea las detecciones al formato esperado por ByteTracker.
        
        Args:
            detections: Lista de detecciones [x_min, y_min, x_max, y_max, conf, clase, movement]
            conf_threshold: Umbral mínimo de confianza
            
        Returns:
            Objeto Detections formateado o None si no hay detecciones válidas
        """
        if len(detections) > 0:
            xywh = []
            # Convertir de formato esquinas a formato centro+dimensiones
            for detection in detections:
                x_min, y_min, x_max, y_max, conf, clase, movement = detection
                cx = (x_min + x_max) / 2    # Centro X
                cy = (y_min + y_max) / 2    # Centro Y
                w = x_max - x_min           # Ancho
                h = y_max - y_min           # Alto
                xywh.append([cx, cy, w, h])

            # Crear arrays numpy para eficiencia
            xywh = np.array(xywh)
            confs = np.array([det[4] for det in detections])
            classes = np.array([det[5] for det in detections])
            movements = np.array([det[6] for det in detections])

            # Crear objeto de detecciones formateado
            formatted_detections = TrackerWrapper.Detections(
                boxes=xywh,
                confidences=confs,
                class_ids=classes,
                movements=movements
            )

            return formatted_detections

        return None

    def track_objects(self, dets_for_tracker, frame):
        """
        Pipeline completo de tracking: procesa detecciones y devuelve tracks con movimiento.
        """
        if dets_for_tracker is None or not hasattr(dets_for_tracker, 'xywh') or len(dets_for_tracker.xywh) == 0:
            print("No hay detecciones válidas para procesar.")
            return []
        
        try:
            # Realizar tracking
            tracks = self.track(dets_for_tracker, frame)
            # Asociar información de movimiento
            tracks_with_movement = self.associate_movements(tracks, dets_for_tracker)
            return tracks_with_movement
        except Exception as e:
            print(f"Error durante el seguimiento: {e}")
            return []

    def visualize(self, frame_original, tracks, original_size, padding_640, target_size):
        """
        Visualiza los tracks en el frame original, convirtiendo coordenadas del espacio
        de procesamiento (640x640) al espacio original de la imagen.
        """
        extra_info_tracks = []
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, clas_id, mov = track

            # Convertir coordenadas del espacio 640x640 al espacio original
            x1_orig, y1_orig, x2_orig, y2_orig = reverse_preprocess_coords(
                x1, y1, x2, y2,
                original_size=original_size,
                padding=padding_640,
                target=target_size
            )

            # Dibujar rectángulo del track
            cv2.rectangle(frame_original,
                          (x1_orig, y1_orig),
                          (x2_orig, y2_orig),
                          (255, 0, 0), 2)

            # Mostrar ID del track
            cv2.putText(frame_original,
                        f"ID: {int(track_id)}",
                        (x1_orig, y1_orig - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2)

            # Calcular centro para posterior cálculo de profundidad
            x_central = (x1_orig + x2_orig) // 2
            y_central = (y1_orig + y2_orig) // 2
            extra_info_tracks.append((x_central, y_central, track_id, conf, clas_id, mov))

        return extra_info_tracks, frame_original

# ======================== MODELO DEPTH ANYTHING V2 ========================
class Depth_Anything_V2:
    """
    Wrapper para el modelo Depth Anything V2 optimizado con TensorRT.
    Estima profundidad monocular de imágenes en tiempo real.
    """

    def __init__(self, gpu_manager: GPUResourceManager):
        """Inicializa el modelo con recursos GPU compartidos."""
        self.gpu_manager = gpu_manager
        self.engine = None
        self.context = None
        self.input_shape = None
        self.output_shape = None
        self.input_mem = None      # Memoria GPU para entrada
        self.output_mem = None     # Memoria GPU para salida
        self.stream = gpu_manager.stream  # Stream CUDA compartido

    def load_engine(self, trt_path):
        """Carga el motor TensorRT del modelo de profundidad."""
        self.engine = self.gpu_manager.load_trt_engine(trt_path)

    def setup_inference(self):
        """
        Configura el entorno de inferencia, reservando memoria en GPU.
        Se ejecuta una sola vez al inicio para optimizar rendimiento.
        """
        self.context = self.engine.create_execution_context()

        # Obtener dimensiones de entrada y salida
        self.input_shape = self.engine.get_binding_shape(0)   # Típicamente (1, 3, 518, 518)
        self.output_shape = self.engine.get_binding_shape(1)  # Típicamente (1, 518, 518)

        # Reservar memoria GPU para datos en precisión FP16 (menor uso de memoria)
        self.input_mem = cuda.mem_alloc(trt.volume(self.input_shape) * np.float16().nbytes)
        self.output_mem = cuda.mem_alloc(trt.volume(self.output_shape) * np.float16().nbytes)
        self.stream = cuda.Stream()

    def infer(self, img):
        """
        Realiza inferencia de profundidad en una imagen.
        
        Args:
            img: Imagen preprocesada de entrada
            
        Returns:
            Tupla (mapa_profundidad_métrico, mapa_profundidad_visual)
        """
        # Preprocesar imagen para el modelo
        img = self.preprocess(img)
        img = img.astype(np.float16)  # Convertir a FP16 para mayor eficiencia

        # Pipeline de inferencia asíncrona en GPU
        cuda.memcpy_htod_async(self.input_mem, img, self.stream)  # CPU → GPU
        self.context.execute_async_v2([int(self.input_mem), int(self.output_mem)], self.stream.handle, None)  # Inferencia
        
        # Recuperar resultado
        output = np.empty(self.output_shape, dtype=np.float16)
        cuda.memcpy_dtoh_async(output, self.output_mem, self.stream)  # GPU → CPU
        self.stream.synchronize()  # Esperar finalización

        # Postprocesar para dos tipos de salida
        return self.postprocess_metric(output), self.postprocess_visual(output)

    def preprocess(self, img):
        """
        Preprocesa imagen para el modelo de profundidad.
        Normaliza valores de píxel y reorganiza dimensiones.
        """
        img = img.astype(np.float16) / 255.0  # Normalizar [0, 255] → [0, 1]
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # HWC → NCHW (1, 3, 518, 518)
        return np.ascontiguousarray(img)  # Optimizar para acceso de memoria

    def postprocess_visual(self, output):
        """
        Postprocesa salida para visualización (imagen en escala de grises).
        """
        output = output.squeeze()  # Eliminar dimensiones batch
        
        # Validar salida
        if np.all(output == 0) or np.isnan(output).any():
            print("⚠️ Salida posiblemente inválida: solo ceros o NaNs")
        
        # Normalizar a rango [0, 1] y convertir a uint8
        output = (output - output.min()) / (output.max() - output.min())
        output = (output * 255).astype(np.uint8)
        
        return output
        
    def postprocess_metric(self, output):
        """
        Postprocesa salida para mediciones métricas de distancia.
        Factor de escala específico del modelo entrenado.
        """
        output = output * 5  # Factor de escala del modelo
        return output

    def visualize(self, depth_map, coords_raw_image, padding_518):
        """
        Crea visualización colorizada del mapa de profundidad y lo restaura
        al tamaño original de la imagen.
        """
        # Normalizar y aplicar mapa de colores
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        # Restaurar al tamaño original eliminando padding
        restored_frame = reverse_preprocess_frame(depth_colormap, coords_raw_image, padding_518, 0.5)
        return restored_frame

    def get_track_depths(self, centers, depth_map, relation_640_518):
        """
        Extrae valores de profundidad para las posiciones centrales de los tracks.
        
        Args:
            centers: Lista de centros de tracks [(x, y, id, ...)]
            depth_map: Mapa de profundidad 2D
            relation_640_518: Factor de conversión entre espacios de coordenadas
            
        Returns:
            Lista de tuplas (profundidad, track_id)
        """
        print(f"Tipo de depth_map: {type(depth_map)}")
        print(f"Forma de depth_map: {depth_map.shape}")

        # Manejar dimensiones de salida del modelo
        if depth_map.ndim == 3 and depth_map.shape[0] == 1:
           depth_map = depth_map[0]  # Reducir de (1, H, W) a (H, W)

        track_depths = []
        for center in centers:
            x, y, id_track = center[:3]
            
            # Convertir coordenadas del espacio 640x640 al espacio 518x518
            x = x / relation_640_518
            y = y / relation_640_518

            # Asegurar que las coordenadas estén dentro de los límites
            x = int(np.clip(x, 0, depth_map.shape[1] - 1))
            y = int(np.clip(y, 0, depth_map.shape[0] - 1))

            # Extraer valor de profundidad en la posición
            depth = float(depth_map[y, x])
            track_depths.append((depth, id_track))

        return track_depths

# ======================== MODELO YOLO OPTIMIZADO ========================
class YOLO:
    """
    Wrapper para modelo YOLO optimizado con TensorRT.
    Realiza detección de objetos en tiempo real con alta eficiencia.
    """
    
    def __init__(self, gpu_manager, device):
        """Inicializa el modelo YOLO con recursos GPU compartidos."""
        self.gpu_manager = gpu_manager
        self.device = device
        self.engine = None
        self.binding_addrs = {}    # Direcciones de memoria para bindings
        self.context = None
        self.bindings = {}         # Información de tensores de entrada/salida

    def load_engine(self, trt_engine_path):
        """Carga el motor TensorRT con plugins necesarios."""
        trt.init_libnvinfer_plugins(self.gpu_manager.logger, namespace="")
        self.engine = self.gpu_manager.load_trt_engine(trt_engine_path)

    def setup_inference(self):
        """
        Configura runtime TensorRT y asigna memoria para inferencias.
        Incluye warmup para optimizar la primera inferencia.
        """
        # Definir estructura para bindings
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        # Configurar cada tensor (entrada y salidas)
        for index in range(self.engine.num_bindings):
            name = self.engine.get_tensor_name(index)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))

            # Manejar dimensiones dinámicas o inválidas
            if any(dim < 0 for dim in shape):
                print(f"Advertencia: El tensor '{name}' tiene dimensiones dinámicas o negativas: {shape}")
                shape = tuple([dim if dim > 0 else 1 for dim in shape])

            # Crear tensor en GPU
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

        self.bindings = bindings
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        self.context = self.engine.create_execution_context()

        # Warmup: ejecutar algunas inferencias dummy para optimizar
        print("Realizando warmup del modelo YOLO...")
        for _ in range(10):
            tmp = torch.randn(1, 3, 640, 640).to(self.device)
            self.binding_addrs['images'] = int(tmp.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))

    def preprocess_image(self, img):
        """
        Preprocesa imagen para inferencia YOLO.
        Convierte formato y normaliza valores de píxel.
        """
        image = img.transpose((2, 0, 1))            # HWC → CHW
        image = np.expand_dims(image, 0)            # Agregar dimensión batch
        image = np.ascontiguousarray(image).astype(np.float32)
        image = torch.from_numpy(image).to(self.device) / 255.0  # Normalizar
        return image

    def infer(self, input_data):
            """
            Ejecuta inferencia en imagen y devuelve detecciones procesadas.

            Args:
                input_data: Tensor de imagen preprocesada

            Returns:
                list: Lista de detecciones [x_min, y_min, x_max, y_max, score, class_id]
            """
            # Actualizar puntero de memoria para la imagen de entrada
            self.binding_addrs['images'] = int(input_data.data_ptr())

            # Ejecutar inferencia
            self.context.execute_v2(list(self.binding_addrs.values()))

            # Extraer resultados de los tensores de salida
            nums = self.bindings['num_dets'].data      # Número de detecciones
            boxes = self.bindings['det_boxes'].data    # Coordenadas de cajas
            scores = self.bindings['det_scores'].data  # Puntuaciones de confianza
            classes = self.bindings['det_classes'].data # IDs de clase

            # Procesar solo las detecciones válidas
            count = nums[0][0] if isinstance(nums, torch.Tensor) else nums[0]
            boxes = boxes[0, :count]
            scores = scores[0, :count]
            classes = classes[0, :count]

            # Formatear detecciones para uso posterior
            detections = []
            for box, score, cl in zip(boxes, scores, classes):
                x_min, y_min, x_max, y_max = box.tolist()
                detections.append([x_min, y_min, x_max, y_max, float(score), int(cl)])

            return detections


    # ===================== MODELO FASTFLOWNET PARA FLUJO ÓPTICO =====================
    class FastFlowNet_model:
        """
        Implementación de FastFlowNet para estimación de flujo óptico.
        Detecta movimiento entre frames consecutivos y mejora la precisión de detección.
        """

        def __init__(self, device, target_size):
            """Inicializa el modelo FastFlowNet."""
            self.device = device
            self.target_size = target_size
            self.model = None
            self.transform = None

        def load_fastflownet(self, model_path):
            """
            Carga el modelo FastFlowNet preentrenado.
            Configura transformaciones para preprocesamiento de imágenes.
            """
            self.model = FastFlowNet().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            # Transformaciones para preprocesamiento
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])

        def process_frame(self, frame1, frame2):
            """
            Calcula flujo óptico entre dos frames consecutivos.

            Args:
                frame1, frame2: Frames consecutivos en formato numpy array

            Returns:
                numpy.ndarray: Flujo óptico [dx, dy, H, W]
            """
            # Convertir frames a tensores
            tensor1 = self.transform(Image.fromarray(frame1)).to(self.device)
            tensor2 = self.transform(Image.fromarray(frame2)).to(self.device)

            # Concatenar para entrada del modelo
            concatenated = torch.cat([tensor1, tensor2], dim=0).unsqueeze(0)

            # Inferencia sin gradientes para eficiencia
            with torch.no_grad():
                flow = self.model(concatenated)

            return flow[0].cpu().numpy()

        def subtract_frame_median(self, flow):
            """
            Elimina movimiento global de cámara restando la mediana.
            Permite detectar solo movimiento relativo de objetos.

            Args:
                flow: Flujo óptico raw

            Returns:
                numpy.ndarray: Flujo óptico con movimiento de cámara eliminado
            """
            median_x = np.median(flow[0])
            median_y = np.median(flow[1])
            return np.array([flow[0] - median_x, flow[1] - median_y])

        def calculate_residual_movement(self, flow):
            """
            Calcula porcentaje de movimiento residual en toda la imagen.
            Útil para detectar escenas con mucho movimiento general.

            Args:
                flow: Flujo óptico procesado

            Returns:
                float: Porcentaje de movimiento (0-100)
            """
            mag, _ = cv2.cartToPolar(flow[0], flow[1])
            total_movement = np.sum(mag)
            num_pixels = flow.shape[1] * flow.shape[2]
            return (total_movement / num_pixels) * 100

        def visualize(self, flow, coords_raw_image, padding_640):
            """
            Genera visualización HSV del flujo óptico.
            Colores indican dirección y intensidad del movimiento.

            Args:
                flow: Flujo óptico a visualizar
                coords_raw_image: Dimensiones imagen original
                padding_640: Padding aplicado durante preprocesamiento

            Returns:
                numpy.ndarray: Imagen de flujo óptico restaurada a tamaño original
            """
            # Crear imagen HSV para visualización
            hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
            mag, ang = cv2.cartToPolar(flow[0], flow[1])

            # Mapear ángulo a matiz (hue) y magnitud a valor (value)
            hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # Hue
            hsv[..., 1] = 255                                        # Saturación máxima
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Convertir HSV a BGR para visualización
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Redimensionar de 160x160 a 640x640
            flow_bgr = cv2.resize(flow_bgr, TARGET_SIZE_640)

            # Restaurar a tamaño original eliminando padding
            restored_frame = reverse_preprocess_frame(flow_bgr, coords_raw_image, padding_640, 0.5)

            return restored_frame

        def detect_movement_in_detection_bbox(self, residual_flow, bbox):
            """
            Detecta movimiento específico dentro de un bounding box.
            Cuantifica qué tanto se mueve un objeto detectado.

            Args:
                residual_flow: Flujo óptico con movimiento de cámara eliminado
                bbox: Coordenadas del bounding box [x_min, y_min, x_max, y_max]

            Returns:
                float: Ratio de movimiento normalizado (0-0.2)
            """
            x_min, y_min, x_max, y_max = map(int, bbox)

            # Obtener dimensiones de flujo e imagen
            flow_h, flow_w = residual_flow.shape[1:3]
            img_h, img_w = self.target_size

            # Escalar bounding box al tamaño del flujo óptico
            x_min = int(x_min * (flow_w / img_w))
            y_min = int(y_min * (flow_h / img_h))
            x_max = int(x_max * (flow_w / img_w))
            y_max = int(y_max * (flow_h / img_h))

            # Asegurar coordenadas dentro de límites
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(flow_w, x_max), min(flow_h, y_max)

            # Calcular magnitud del movimiento
            mag, _ = cv2.cartToPolar(residual_flow[0], residual_flow[1])

            # Extraer región de interés
            bbox_flow = mag[y_min:y_max, x_min:x_max]

            if bbox_flow.size == 0:
                return 0

            # Calcular movimiento normalizado por área
            total_movement = np.sum(bbox_flow)
            bbox_total_area = (x_max - x_min) * (y_max - y_min)
            movement_ratio = min(0.2, (total_movement * 3 / bbox_total_area))

            return movement_ratio

        def calculate_movement_and_filter(self, frame, detections, residual_flow):
            """
            Ajusta confianza de detecciones basado en movimiento y filtra resultados.
            Las detecciones con movimiento obtienen mayor confianza.

            Args:
                frame: Frame actual (para debug opcional)
                detections: Lista de detecciones YOLO
                residual_flow: Flujo óptico procesado

            Returns:
                list: Detecciones filtradas con movimiento incluido
            """
            adjusted_detections = []

            for detection in detections:
                x_min, y_min, x_max, y_max, conf, clase = detection

                # Calcular movimiento en el bounding box
                movement = self.detect_movement_in_detection_bbox(
                    residual_flow, (x_min, y_min, x_max, y_max)
                )

                # Ajustar confianza sumando movimiento
                conf += movement
                conf = min(conf, 1.0)  # Limitar a máximo 1.0

                # Filtrar solo detecciones con confianza suficiente
                if conf > TRACKER_CONF_THRESHOLD:
                    adjusted_detections.append([x_min, y_min, x_max, y_max, conf, clase, movement])

            return adjusted_detections


    # ===================== FUNCIONES DE PREPROCESAMIENTO =====================

    def preprocess_frame(frame, target_size):
        """
        Redimensiona frame manteniendo aspect ratio y añade padding.
        Esencial para modelos que requieren tamaño de entrada fijo.

        Args:
            frame: Imagen de entrada
            target_size: Tamaño objetivo (height, width)

        Returns:
            tuple: (frame_padded, padding_info)
                - frame_padded: Imagen redimensionada con padding
                - padding_info: (top, bottom, left, right) para reversión
        """
        h, w = frame.shape[:2]

        # Calcular escala manteniendo aspect ratio
        scale = min(target_size[1] / w, target_size[0] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Redimensionar imagen
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calcular padding para centrar imagen
        top    = (target_size[0] - new_h) // 2
        bottom = target_size[0] - new_h - top
        left   = (target_size[1] - new_w) // 2
        right  = target_size[1] - new_w - left

        # Aplicar padding con borde negro
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        padding = (top, bottom, left, right)
        return padded, padding


    def reverse_preprocess_frame(processed_frame, original_size, padding, output_scale=1.0):
        """
        Revierte el preprocesamiento eliminando padding y reescalando.
        Restaura imagen a dimensiones originales o escaladas.

        Args:
            processed_frame: Imagen procesada con padding
            original_size: (height, width) imagen original
            padding: (top, bottom, left, right) padding aplicado
            output_scale: Factor de escala final (default: 1.0)

        Returns:
            numpy.ndarray: Imagen restaurada
        """
        print("Tamaño entrada:", processed_frame.shape)

        top, bottom, left, right = padding

        # 1. Eliminar padding
        cropped = processed_frame[top:processed_frame.shape[0] - bottom,
                                left:processed_frame.shape[1] - right]
        if cropped.size == 0:
            raise ValueError(
                f"Cropped vacío: top={top}, bottom={bottom}, left={left}, right={right}"
            )

        # 2. Reescalar a tamaño original
        orig_h, orig_w = original_size
        restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        print("Tamaño medio:", restored.shape)

        # 3. Aplicar escala final opcional
        if output_scale != 1.0:
            final_w = max(1, int(orig_w * output_scale))
            final_h = max(1, int(orig_h * output_scale))
            restored = cv2.resize(restored, (final_w, final_h), interpolation=cv2.INTER_LINEAR)

        print("Tamaño salida:", restored.shape)

        return restored


    def reverse_preprocess_coords(x1, y1, x2, y2, original_size, padding, target):
        """
        Convierte coordenadas desde imagen preprocesada a imagen original.
        Esencial para mapear detecciones correctamente.

        Args:
            x1, y1, x2, y2: Coordenadas en imagen preprocesada
            original_size: (height, width) imagen original
            padding: (top, bottom, left, right) padding aplicado
            target: (height, width) tamaño objetivo usado

        Returns:
            tuple: (x1_orig, y1_orig, x2_orig, y2_orig) coordenadas originales
        """
        orig_h, orig_w = original_size
        top, bottom, left, right = padding

        # Paso 1: Quitar offset del padding
        x1_nopad = x1 - left
        x2_nopad = x2 - left
        y1_nopad = y1 - top
        y2_nopad = y2 - top

        # Paso 2: Revertir escala aplicada
        scale = min(target[1] / orig_w, target[0] / orig_h)
        x1_orig = int(x1_nopad / scale)
        x2_orig = int(x2_nopad / scale)
        y1_orig = int(y1_nopad / scale)
        y2_orig = int(y2_nopad / scale)

        return x1_orig, y1_orig, x2_orig, y2_orig


    # ===================== SCRIPT PRINCIPAL =====================

    if __name__ == "__main__":
        """
        Script principal que orquesta todo el sistema de detección.
        Procesa video frame por frame aplicando todos los modelos en secuencia.
        """

        # ===== CONFIGURACIÓN INICIAL =====
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        opt = parser.parse_args()

        device = select_device(opt.device)

        # Diccionario de mapeo de clases
        class_names = {
            0: "Infantry",
            1: "Tank",
            2: "Mechanized Inf"
        }

        # ===== INICIALIZACIÓN DE MODELOS =====
        print("🚀 Inicializando sistema de detección...")

        # Administrador de recursos GPU compartidos
        gpu_manager = GPUResourceManager()

        # Modelo de estimación de profundidad
        print("📏 Cargando Depth Anything V2...")
        depth_anything_v2 = Depth_Anything_V2(gpu_manager)
        depth_anything_v2.load_engine(TRT_ENGINE_PATH_DEPTH_ANYTHING_V2)
        depth_anything_v2.setup_inference()

        # Modelo de detección de objetos
        print("🎯 Cargando YOLO...")
        yolo = YOLO(gpu_manager, device)
        yolo.load_engine(TRT_ENGINE_PATH_YOLO)
        yolo.setup_inference()

        # Modelo de flujo óptico
        print("🌊 Cargando FastFlowNet...")
        fastflownet = FastFlowNet_model(device, TARGET_SIZE_640)
        fastflownet.load_fastflownet(FASTFLOWNET_PATH)

        # Sistema de seguimiento
        print("🔍 Inicializando tracker...")
        tracker = TrackerWrapper(frame_rate=30)

        # ===== CONFIGURACIÓN DE VIDEO =====
        cap = cv2.VideoCapture(VIDEO_PATH)

        # Leer primer frame para configuración inicial
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: No se pudo leer el primer frame.")
            cap.release()
            exit()

        # Configuración inicial basada en dimensiones del frame
        prev_frame_resized, _ = preprocess_frame(frame, TARGET_SIZE_640)
        w_raw_image, h_raw_image, _ = frame.shape
        coords_raw_image = (w_raw_image, h_raw_image)

        # ===== CONFIGURACIÓN DE INTERFAZ =====
        print("🖥️  Inicializando interfaz gráfica...")
        root = tk.Tk()
        app = DetectionApp(root, w_raw_image, h_raw_image)

        # Esperar renderizado completo de GUI
        root.update_idletasks()
        root.update()
        time.sleep(1)

        # ===== BUCLE PRINCIPAL DE PROCESAMIENTO =====
        print("🎬 Iniciando procesamiento de video...")
        while cap.isOpened():
            frame_start_time = time.time()

            # Leer siguiente frame
            ret, frame = cap.read()
            if not ret:
                break

            # ===== PREPROCESAMIENTO =====

            # Preprocesar para diferentes tamaños según modelo
            frame_resized_640, padding_640 = preprocess_frame(frame, TARGET_SIZE_640)
            frame_resized_518, padding_518 = preprocess_frame(frame, TARGET_SIZE_518)

            # ===== DETECCIÓN CON YOLO =====

            # Preparar datos para TensorRT
            input_data_yolo = yolo.preprocess_image(frame_resized_640)

            # Inferencia YOLO
            detections = yolo.infer(input_data_yolo)

            # ===== ANÁLISIS DE MOVIMIENTO =====

            # Conversión de formato de color
            prev_frame_rgb = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.cvtColor(frame_resized_640, cv2.COLOR_BGR2RGB)

            # Cálculo de flujo óptico
            flow = fastflownet.process_frame(prev_frame_rgb, frame_rgb)

            # Postprocesamiento del flujo
            residual_flow = fastflownet.subtract_frame_median(flow)
            prev_frame_resized = frame_resized_640.copy()
            movement_percentage = fastflownet.calculate_residual_movement(residual_flow)

            # Eliminar flujo si hay demasiado movimiento general
            if movement_percentage >= 10:
                residual_flow[:] = 0


            # Ajustar confianza basado en movimiento
            adjusted_detections = fastflownet.calculate_movement_and_filter(
                frame_resized_640, detections, residual_flow
            )

            # Generar visualización de flujo
            visualize_frame_fastflownet = fastflownet.visualize(
                residual_flow, coords_raw_image, padding_640
            )

            # ===== SEGUIMIENTO DE OBJETOS =====

            # Formatear detecciones para tracker
            dets_for_tracker = tracker.format_detections_for_tracker(
                adjusted_detections, TRACKER_CONF_THRESHOLD
            )

            # Ejecutar seguimiento
            tracks = tracker.track_objects(dets_for_tracker, frame_resized_640)

            # Visualizar resultados de tracking
            extra_info_tracks, visualize_frame_byte_track = tracker.visualize(
                frame, tracks, coords_raw_image, padding_640, TARGET_SIZE_640
            )

            # ===== ESTIMACIÓN DE PROFUNDIDAD =====

            # Inferencia de profundidad
            depth_map, depth_map_color = depth_anything_v2.infer(frame_resized_518)

            # Visualización y cálculo de distancias
            visualize_frame_depth_anything_v2 = depth_anything_v2.visualize(
                depth_map_color, coords_raw_image, padding_518
            )
            track_depths = depth_anything_v2.get_track_depths(
                extra_info_tracks, depth_map, RELATION_640_518
            )

            # ===== INTEGRACIÓN DE DATOS =====

            # Crear diccionario de distancias por track ID
            depth_dict = {int(track_id): distancia for distancia, track_id in track_depths}

            # Combinar toda la información por track
            info_data = [
                (
                    int(track_id),                    # ID del track
                    class_names.get(clas_id),        # Nombre de la clase
                    conf * 100,                      # Confianza en porcentaje
                    (mov * 10000) / 20,              # Movimiento normalizado
                    depth_dict[int(track_id)]        # Distancia estimada
                )
                for _, _, track_id, conf, clas_id, mov in extra_info_tracks
                if int(track_id) in depth_dict
            ]

            # ===== ACTUALIZACIÓN DE INTERFAZ =====

            # Actualizar todos los paneles de la GUI
            app.update_detection_image(visualize_frame_byte_track)
            app.update_motion_image(visualize_frame_fastflownet)
            app.update_depth_image(visualize_frame_depth_anything_v2)
            app.update_info_panel(info_data)

            # Refrescar interfaz
            app.root.update_idletasks()
            app.root.update()

        # Liberar recursos
        cap.release()

    print("✅ Procesamiento completado exitosamente!")
