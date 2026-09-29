import socket
import threading
from PIL import Image, ImageFile
import io
from ultralytics import YOLO
import torch
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Callback para procesar la imagen
def process_image(image_data, width, height):
    try:
        # Convertir los datos binarios en una imagen usando PIL
        img = Image.frombytes('RGB', (width, height), image_data)
        print(f"Imagen recibida. Tamaño: {img.size}, Modo: {img.mode}")
        # Aquí puedes hacer el procesamiento que necesites en la imagen.
        # Por ejemplo, guardar la imagen para ver el resultado:
        img.save("received_image.bmp")
        return "Procesamiento completado"
    except Exception as e:
        return f"Error al procesar la imagen: {e}"

def recv_all(sock, size):
    buffer = b''
    while len(buffer) < size:
        chunk = sock.recv(size - len(buffer))
        if not chunk:
            raise ConnectionError("Conexión cerrada inesperadamente")
        buffer += chunk
    return buffer

# Función para manejar cada conexión de cliente
def handle_client_connection(conn, addr, model):
    print(f"Conexión recibida de {addr}")

    try:
        while True:
            size_data = conn.recv(4)  # Recibimos los 4 bytes que indican el tamaño
            if not size_data:
                print("Error: No se pudo recibir el tamaño de la imagen.")
                return

            image_size = int.from_bytes(size_data, byteorder='little')

            print(f"Tamaño de la imagen recibido: {image_size} bytes")

            # Recibir los datos de la imagen
            image_data = recv_all(conn, image_size)
            if not image_data:
                print("Error: No se pudo recibir los datos de la imagen.")
                return

            print("Imagen recibida con éxito.")

            im = Image.open(io.BytesIO(image_data))

            results = model(im, device="cuda", conf=0.9)
            n_hits = len(results[0].masks)

            if n_hits > 0:
                results[0].show()
                hit_info = []

                #for box in results[0].boxes:
                #    cords = box.xywh.cpu().numpy()[0]
                #    hit_info.append([cords[0], cords[1], 0, box.conf.item()])

                for m in results[0].masks:
                    area = torch.sum(m.data).item()
                    if(area > 20000):
                        mask = m.data.cpu().numpy().squeeze()  # from (1, 640, 1, 512) → (640, 512)
                        mask_resized = cv2.resize(mask, (784, 1006))  # Camera (width, height)

                        x, y = np.nonzero(mask_resized)
                        c_x = np.mean(x)
                        c_y = np.mean(y)
                        hit_info.append([c_x, c_y, 0, 1])

                hits_str = ""
                for hit in hit_info:
                    hits_str += ",".join(map(str, hit)) + ";"

                print(hits_str)
                conn.sendall(hits_str.encode('utf-8'))
            else:
                hits_str = ";"
                conn.sendall(hits_str.encode('utf-8'))

    finally:
        conn.close()


# Funcion para escuchar las conexiones entrantes
def start_server(host="192.168.88.11", port=5000):
    print("Loading yolo model...")
    model = YOLO('pat-papercup-seg.pt')  # Modelo pequeño para detección rápida
    model.to("cuda")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Servidor escuchando en {host}:{port}")

    while True:
        conn, addr = server.accept()
        # Manejar la conexión de cada cliente en un hilo separado
        threading.Thread(target=handle_client_connection, args=(conn, addr, model)).start()


if __name__ == "__main__":
    while True:
        start_server()
