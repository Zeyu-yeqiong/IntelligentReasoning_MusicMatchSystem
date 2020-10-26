import os
import cv2
import time


def extract_frame(filename):
    filepath = "/static/images/" + filename + '/' + str(time.time())
    os.mkdir(filepath)
    vidcap = cv2.VideoCapture(filepath)
    success, image = vidcap.read()
    count = 0
    # print(success)
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(frame_count,frame_count//6)
    res = []
    for i in range(5):
        res.append(frame_count // 6 * i)
    print(res)
    count_img = 0
    while success:
        success, image = vidcap.read()
        if count in res:
            filepath.append()
            cv2.imwrite(filepath + '/' + str(count_img) + '.jpg', image)  # save frame as JPEG file
            count_img += 1
        count += 1
    return filepath