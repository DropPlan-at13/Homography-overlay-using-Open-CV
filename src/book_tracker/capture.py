import cv2


def open_camera(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    return frame
