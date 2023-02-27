from pathlib import Path
from cvu.detector.yolov5 import Yolov5 as Yolov5Onnx
import polars as pl
import paramiko
import time
import cv2

# FILE = Path(__file__)
# FILE = Path('./main.py').absolute()
# DATA_DIR = FILE.parent / 'data'
DATA_DIR = Path('./data')
NAME_PATH = Path('./models') / 'yolo.names'
TEST_PATH = DATA_DIR / 'test.jpg'

def get_classes(path=NAME_PATH):
    return pl.read_csv(path, has_header=False)['column_1'].to_list()

COLORS = np.random.uniform(0, 255, size=(len(get_classes()), 3))

def pltimg(img):
    return plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def capture_and_fetch(local_path='./data/capture.jpg', delay=None):
    HOSTNAME="192.168.1.13"
    username='pi'
    # TODO: replace with env var or sshkey
    password=None

    ssh_client =paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=HOSTNAME,username=username,password=password)

    if delay:
        time.sleep(delay)

    cmd = "raspistill -t 0 -h 640 -w 640 -o ~/Desktop/capture.jpg"
    stdin,stdout,stderr=ssh_client.exec_command(cmd)
    time.sleep(1)

    ftp_client=ssh_client.open_sftp()
    ftp_client.get('/home/pi/Desktop/capture.jpg',local_path)
    ftp_client.close()

def predict(model, image_path):
    img = cv2.imread(image_path)
    preds = model(img)
    fig = pltimg(preds.draw(img))
    fig.write_png(image_path[:-4] + '_yolo.png')
    return preds, img


model = Yolov5Onnx(classes="coco",
                       backend="onnx",
                       weight='yolov5s',
                       device='cpu')

