import cv2
import multiprocessing
import time


def main_process(input, output):
    while True:
        time.sleep(2)
        image = input.get()
        output.put(image)


def child_process(input, output):
    while True:
        image = input.get()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 100)
        output.put(edges)


if __name__ == '__main__':
    input = multiprocessing.Queue(1)
    output = multiprocessing.Queue(1)

    main_process = multiprocessing.Process(target=main_process, args=(input, output))
    main_process.daemon = True
    child_process = multiprocessing.Process(target=child_process, args=(input, output))
    child_process.daemon = False

    main_process.start()
    child_process.start()

    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()

        input.put(frame)

        cv2.imshow('Video', output.get())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
