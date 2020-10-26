import cv2
import os
import datetime


def extract_frame(uploadpath):
    # print(filename)
    vidcap = cv2.VideoCapture(uploadpath)
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

    current=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filepath = "./static/images/" + current
    os.mkdir(filepath)
    while success:
        success, image = vidcap.read()
        if count in res:
            cv2.imwrite(filepath + '/' + str(count_img) + '.jpg', image)  # save frame as JPEG file
            # print(filepath + '/' + str(count_img) + '.jpg')
            count_img += 1
        count += 1
    return filepath
