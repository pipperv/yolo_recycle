import socket
import random

HOST = '192.168.88.11'  # Dirección del servidor
PORT = 5000        # Puerto que debe coincidir con el usado en ConnectToSocket

def generate_random_hit():
    x = round(random.uniform(0, 100), 2)
    y = round(random.uniform(0, 100), 2)
    angle = round(random.uniform(0, 360), 2)
    score = round(random.uniform(0.5, 1.0), 2)
    return f"{x} {y} {angle} {score}"

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[INFO] Servidor escuchando en {HOST}:{PORT}...")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"[INFO] Conexión aceptada desde {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    message = data.decode('utf-8').strip()
                    print(f"[RECV] {message}")

                    # if message.upper() == "ANALYZE":
                    #     response = generate_random_hit() + "\n"
                    #     print(f"[SEND] {response.strip()}")
                    #     conn.sendall(response.encode('utf-8'))

if __name__ == "__main__":
    start_server()