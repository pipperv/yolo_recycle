import socket
import threading
from PIL import Image, ImageFile
import io
from ultralytics import YOLO
import time

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

    with open("record/last.txt", "r") as last:
        img_count = int(last.read())
        print("Starting recording from "+str(img_count))

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
            im = im.convert("RGB")

            img_w, img_h = im.size

            results = model(im, device="cuda", conf=0.9)
            n_hits = len(results[0].boxes)

            if n_hits > 0:
                im.save("record/labeled/images/"+str(img_count)+".jpg")
                file = open("record/labeled/labels/"+str(img_count)+".txt", "w")
                hit_info = []
                text = ''

                img_count += 1

                for box in results[0].boxes:
                    cords = box.xywh.cpu().numpy()[0]
                    hit_info.append([0, cords[0], cords[1], cords[2], cords[3]])
                    line = ' '.join([str(x) for x in [0, cords[0]/img_w, cords[1]/img_h, cords[2]/img_w, cords[3]/img_h]])
                    text = text + line + "\n"
                print(text)
                file.write(text[:-2])
                file.close()

            else:
                im.save("record/background/images/" + str(img_count) + ".jpg")
                img_count += 1

            with open("record/last.txt", "w") as last:
                last.write(str(img_count))

            hits_str = ";"
            conn.sendall(hits_str.encode('utf-8'))

    finally:
        conn.close()


# Funcion para escuchar las conexiones entrantes
def start_server(host="192.168.88.11", port=5000):
    print("Loading yolo model...")
    model = YOLO('best.pt')  # Modelo pequeño para detección rápida
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
