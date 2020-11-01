# coding:utf-8
 
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import subprocess
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
import logging
from logging.handlers import TimedRotatingFileHandler
from werkzeug.utils import secure_filename
import os
import cv2
import time
import re
import ruleEngine1
from datetime import timedelta
import demo_coco_gcn
import img_multi_label as img_read
import detection
 
#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp','avi', 'mp4','AVI', 'MP4'])

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)

app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/', methods=['POST', 'GET'])
def redirect_default():
    print(request.path)
    return redirect(request.url+"index",code=302)

 
 
@app.route('/index', methods=['POST', 'GET']) 
def upload():
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请pls+png、PNG、jpg、JPG、bmp"})
        

        #change start------------------------------
        upload_file=f.filename
        basepath = os.path.dirname(__file__)
        print('basepath:', basepath)
        file_format=upload_file[upload_file.rfind('.')+1:]
        if file_format  in ['avi', 'mp4','AVI', 'MP4']:
            upload_path=os.path.join(basepath, 'static/videos', secure_filename(f.filename))#move file to videos folder
            print('upload_path:',upload_path)
            f.save(upload_path)
            image_path = detection.extract_frame(upload_path)
            # print(image_path)
            labels = []
            # 使用Opencv转换一下图片格式和名称
            model, inp, transform_com = img_read.init_model()
            for image in os.listdir(image_path):
                print(image_path + '/' + image)
                img = cv2.imread(image_path + '/' + image)
                cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
                # labels=main_coco(img)

                labelFrame = img_read.seach_label('static\\images\\test.jpg', model, inp, transform_com)
                print(labelFrame)
                for i in range(len(labelFrame)):
                    if labelFrame[i] not in labels:
                        labels.append(labelFrame[i])
            labels = labels[:2]
            # labels = ['person','car']
            # print('labels:', labels)
            music_type = ruleEngine1.find_music(labels)
            if is_number(music_type[-2]):
                dics = {1: 'passion1', 2: 'passion-1', 3: 'excited1', 4: 'excited-1', 5: 'happy1', 6: 'happy-1',
                        7: 'relaxed-1', 8: 'relaxed1', 9: 'quieted-1', 10: 'quiet1'}
                dics_ch = {value: key for key, value in dics.items()}
                music_type_first = music_type[:-1]
                music_first_num = dics_ch[music_type_first]
                music_first_change = abs(music_first_num - int(music_type[-1]))
                music_type = dics[music_first_change]

            print(music_type)
            if music_type[-2] == '-':
                folder = music_type[:-2]
                degree = '-1'
            else:
                folder = music_type[:-1]
                degree = music_type[-1]
            print(music_type, folder)
            dic = {'0': 'med', '1': 'high', '-1': 'light'}
            degree = dic[degree]

            music_path = 'static/music/' + folder + '/' + degree
            print(music_path)
            all_files = os.listdir(music_path)
            # music_path='excited/high/'
            # all_files = os.listdir(basepath + '/static/music/'+music_path)
            #change  end-------------------------------
            
            music_list = []
            for i in all_files:
                x = re.findall(r'(.*?).mp3', i)
                music_list.append(music_path+x[0])

            #extract video
            get_ori_mp3 = 'ffmpeg -i "static/videos/'+f.filename.replace(" ","_")+'" -y "static/videos/temp.mp3"'
            # print(f.filename)
            # print(getmp4)
            combine_a_a='ffmpeg -i "static/videos/temp.mp3" -i "'+music_path+"/"+x[0]+  '.mp3" -filter_complex amerge -c:a libmp3lame -q:a 4  -y audio.mp3'
            print('combine aa:',combine_a_a)
            

            combine_a_v='ffmpeg -i "static/videos/'+f.filename.replace(" ","_")+'" -i audio.mp3 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -t 15.0 -y static/videos/output.mp4 '
            print('combine av:', combine_a_v)
            subprocess.call(get_ori_mp3,shell = True)
            subprocess.call(combine_a_a,shell = True)
            subprocess.call(combine_a_v,shell = True)
            return render_template('index_video.html',val1=time.time(),music_list=music_list)

        else:
            print(file_format,'image')
            user_input = request.form.get("name")
    
            basepath = os.path.dirname(__file__) 
            upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  #filename可以改
            f.save(upload_path)
    
            # 使用Opencv转换一下图片格式和名称
            img = cv2.imread(upload_path)
            cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
            model, inp, transform_com = img_read.init_model()
            labelFrame = img_read.seach_label('static\\images\\test.jpg', model, inp, transform_com)
            print(labelFrame)
            labels = []
            for i in range(len(labelFrame)):
                if labelFrame[i] not in labels:
                    labels.append(labelFrame[i])

            labels = labels[:2]
            # labels = ['person','car']
            # print('labels:', labels)
            music_type = ruleEngine1.find_music(labels)
            if is_number(music_type[-2]):
                dics = {1: 'passion1', 2: 'passion-1', 3: 'excited1', 4: 'excited-1', 5: 'happy1', 6: 'happy-1',
                        7: 'relaxed-1', 8: 'relaxed1', 9: 'quiet-1', 10: 'quiet1'}
                dics_ch = {value: key for key, value in dics.items()}
                music_type_first = music_type[:-1]
                music_first_num = dics_ch[music_type_first]
                music_first_change = abs(music_first_num - int(music_type[-1]))
                music_type = dics[music_first_change]

            print(music_type)
            if music_type[-2] == '-':
                folder = music_type[:-2]
                degree = '-1'
            else:
                folder = music_type[:-1]
                degree = music_type[-1]
            print(music_type, folder)
            dic = {'0': 'med', '1': 'high', '-1': 'light'}
            degree = dic[degree]
            music_path = basepath + '/static/music/' + folder + '/' + degree

            all_files = os.listdir(music_path)
            print(all_files)
            music_list = []
            for i in all_files:
                x = re.findall(r'(.*?).mp3', i)
                music_list.append(x[0])
            #change start -----------------------------------
            # music_path='excited/high/'
            # all_files = os.listdir(basepath + '/static/music/'+music_path)
            #change  end-------------------------------
            
            # music_list = []
            # for i in all_files:
            #     x = re.findall(r'(.*?).mp3', i)
            #     music_list.append(music_path+x[0])

            return render_template('index_ok.html',userinput=user_input,val1=time.time(),music_list=music_list)

    if request.method == 'GET':
        if len(request.args) > 0:
            # watch_time.append(request.args['time']) to do list: logger
            app.logger.info('Watch Time:' + request.args['time'])

    return render_template('index.html')

if __name__ == '__main__':
    # app.debug = True
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] - %(message)s")
    handler = TimedRotatingFileHandler(
    "flask.log", when="D", interval=1, backupCount=30,
    encoding="UTF-8", delay=False, utc=True)
    app.logger.addHandler(handler)
    handler.setFormatter(formatter)
    app.run(host='127.0.0.1', port=8987, debug=True)
