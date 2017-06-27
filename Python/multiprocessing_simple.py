## Simple example for multiprocessing using OpenCV

import cv2
import multiprocessing
from multiprocessing import Process, Queue
import time


def worker(input, output):
    while True:
        image = input.get()
        output.put(image)


if __name__ == '__main__':
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input = Queue(1)
    output = Queue(1)

    process = Process(target=worker, args=((input, output)))
    process.daemon = True
    process.start()  # Launch as a separate python process

    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        time.sleep(2)
        input.put(frame)

        cv2.imshow('Video', output.get())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    process.join()  # Wait for the process to finish

    video_capture.release()
    cv2.DestroyAllWindows()

