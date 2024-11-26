import cv2
from realsense_camera import RealsenseCamera


camera = RealsenseCamera(640, 360, 320, 240)

while True:
    ret, color_image, depth_image = camera.get_frame_stream()
    height, width, _ = color_image.shape
    
    cv2.imshow("Color Image", color_image)
    cv2.imshow("Depth Image", depth_image)

    # check every 1ms if Esc is pressed. When pressed, finish stream
    key = cv2.waitKey(1)
    if key == ord('\x1b'):
        break

camera.stop()
