from models.experimental import attempt_load
from utils.torch_utils import select_device
from PIL import Image
import base64
import io
from flask import Flask, request, jsonify,render_template
import json
import numpy as np
from backend.predict import predict
from pathlib import Path
import  cv2


# 传入__name__实例化Flask
app = Flask(__name__,static_folder='templates/frontend')
# 自动重载模板文件
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# 读取flask配置
with open('./backend/flask_config.json','r',encoding='utf8')as fp:
    opt = json.load(fp)
    print('Flask Config : ', opt)

# 选择设备
device = select_device(opt['device'])
# 加载模型
model = attempt_load(opt['weights'], map_location=device)  


# opencv读取出来的图片相当于numpy数组
def cv2_to_base64(image):
    image1 = cv2.imencode('.jpg', image)[1]
    image_code = str(base64.b64encode(image1))[2:-1]
    return image_code

def base64_to_cv2(image_code):
    #解码
    img_data=base64.b64decode(image_code)
    #转为numpy
    img_array=np.fromstring(img_data,np.uint8)
    #转成opencv可用格式
    img=cv2.imdecode(img_array,cv2.COLOR_RGB2BGR)
    return img


@app.route('/predict/', methods=['POST'])
# 响应POST消息的预测函数
def get_prediction():
    return jsonify()
    response = request.get_json()
    data_str = response['image']
    point = data_str.find(',')
    base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"
    image = base64.b64decode(base64_str) # base64图像解码
    img = Image.open(io.BytesIO(image)) # 打开文件
    if (img.mode != 'RGB'):
        img = img.convert("RGB")
    save_path = str(Path(opt['source']) / Path("img4predict.jpg")) # 保存路径
    img.save(save_path) # 保存文件
    # img.save("./frontend/static/images/img4predict.jpg")  

    # convert to numpy array.
    img_arr = np.array(img)
    # print('img_arr shape = %s \n' % str(img_arr.shape))

    results = predict(opt, model, img_arr) # 预测图像

    return jsonify(results)



@app.route('/', methods=['POST','GET'])
# 响应POST消息的预测函数
def index():
    return render_template('index.html')

@app.route('/postimg/', methods=['POST','GET'])
# 响应POST消息的预测函数
def postimg():


    return render_template('index.html',image_name='frontend/static/output/img4predict.jpg')

    # print(request.files)

    # return request.get_json()
    response = request.get_json()


    # data_str = response['image']
    print(response)
    print(json.loads(response))
    results=json.loads(response)

    return render_template('index.html', warning='request.get_json()')
    # point = data_str.find(',')
    # base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"
    # image = base64.b64decode(base64_str) # base64图像解码
    # img = Image.open(io.BytesIO(image)) # 打开文件
    # if (img.mode != 'RGB'):
    #     img = img.convert("RGB")
    # save_path = str(Path(opt['source']) / Path("img4predict.jpg")) # 保存路径
    # img.save(save_path) # 保存文件
    # img.save("./frontend/static/images/img4predict.jpg")

    # convert to numpy array.
    # img_arr = np.array(img)
    # print('img_arr shape = %s \n' % str(img_arr.shape))

    # results = predict(opt, model, img_arr) # 预测图像


    # print(results)
    return jsonify(results)



@app.after_request
def add_headers(response):
    # 允许跨域
    response.headers.add('Access-Control-Allow-Origin', '*') 
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1',port=5000)
    #app.run(debug=False, host='127.0.0.1')



