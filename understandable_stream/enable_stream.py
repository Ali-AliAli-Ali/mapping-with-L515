# libraries for camera stream
# import pyrealsense2 as rs
import cv2
from realsense_camera import RealsenseCamera
# libraries for sending stream
import socket
import pickle
import struct


# def get_camera_ip():
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.connect(("8.8.8.8", 80))   # -> change to needed ip. 
#     return sock.getsockname()[0]


# host_ip = get_camera_ip()
camera_ip = "172.20.10.10"
print(f"Server ip-address: {camera_ip}")
port = 1024

# Socket creation
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
    server_address = (camera_ip, port)
    server_sock.bind(server_address)
    server_sock.listen(5)  # RS use 10

    print(f"Server started at {server_sock}")

    # camera creation
    camera = RealsenseCamera(640, 360, 320, 240)  # maximum available with USB-2.1

    try:
        while True:
            # accept connection from client
            client, addr = server_sock.accept()

            print(f"Connected to Client@{addr}")
            if client:
                # stream start
                ret, color_image, depth_image = camera.get_frame_stream()
                height, width, _ = color_image.shape

                a = pickle.dumps(color_image, depth_image)
                message = struct.pack("Q", len(a)) + a
                client.sendall(message)  # change to recv?

                cv2.imshow("Color Image", color_image)
                cv2.imshow("Depth Image", depth_image)

                # check every 1ms if Esc is pressed. When pressed, finish
                key = cv2.waitKey(1)
                if key == ord('\x1b'):
                    break

    finally:
        camera.stop()

