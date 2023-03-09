# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_core.ipynb.

# %% auto 0
__all__ = ['FILE_PATH', 'DATA_DIR', 'NAME_PATH', 'TEST_PATH', 'CMD', 'pltimg', 'capture_and_fetch', 'predict',
           'get_default_model', 'run_bot']

# %% ../00_core.ipynb 5
import os
FILE_PATH = os.path.abspath('') 

# %% ../00_core.ipynb 7
import random

import discord
from pathlib import Path
import polars as pl
import paramiko
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
from rich import print

DATA_DIR = Path(FILE_PATH).parent.parent / "data"
NAME_PATH = (DATA_DIR.parent / "models") / "yolo.names"
TEST_PATH = DATA_DIR / "test.jpg"
CMD = "raspistill -t 0 -h 640 -w 640 -o ~/Desktop/capture.jpg"


def pltimg(img: cv2.Mat) -> plt.Figure:
    """Plots picture"""
    return plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def capture_and_fetch(
    username: str = "pi", # Username for Raspberry Pi
    local_path: str = "./data/capture.jpg", # Path to save image
    delay: int = None, # Delay between command and picture is taken
    cmd=CMD, # Raspberry pi camera command
):
    """Requires `PI_PASSWORD be set in .env file. Delay is the number of milliseconds before taking picture"""
    load_dotenv()
    hostname = os.environ['PI_HOSTNAME']
    password = os.environ["PI_PASSWORD"]
    ssh_client = _connect(hostname, username, password)

    if delay:
        time.sleep(delay)

    ssh_client.exec_command(cmd)
    time.sleep(1)

    _fetch(ssh_client, "/home/pi/Desktop/capture.jpg", local_path)


def _connect(hostname: str, username: str, password: str) -> paramiko.SSHClient():
    """Connects to client"""
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=hostname, username=username, password=password)
    return ssh_client


def _fetch(client: paramiko.SSHClient, remote_path: str, local_path: str):
    """Fetches single file using SSHClient"""
    ftp_client = client.open_sftp()
    ftp_client.get(remote_path, local_path)
    ftp_client.close()



# %% ../00_core.ipynb 9
from cvu.detector.yolov5 import Yolov5 as Yolov5Onnx

def predict(model: Yolov5Onnx, image_path: str):
    """Runs model on input image and returns predictions and output image"""
    img = cv2.imread(image_path)
    preds = model(img)
    fig = pltimg(preds.draw(img))
    fig.write_png(image_path[:-4] + "_yolo.png")
    return preds, img


def get_default_model() -> Yolov5Onnx:
    """Retrieves default YOLOv5 model"""
    return Yolov5Onnx(classes="coco", backend="onnx", weight="yolov5s", device="cpu")

# %% ../00_core.ipynb 13
def run_bot():
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        """Indicates bot is ready"""
        print(f"{client.user} has connected to Discord. Hello!")

    @client.event
    async def on_member_join(member):
        """Responds to members joining"""
        await member.create_dm()
        await member.dm_channel.send(
            f"Hello {member.name}, welcome to Islander Walk Securitron 9000"
        )

    @client.event
    async def on_message(message):
        """Responds to messages from users"""
        if message.author == client.user:
            return

        if message.content == "99!":
            b99 = ["Cool. Cool Cool Cool Cool", "no doubt no doubt no doubt"]
            response = random.choice(b99)
            await message.channel.send(response)

        elif message.content.lower() == "pi!":
            await take_picture(message)

        elif message.content.lower() == 'oddpi!':
            await take_and_model_picture(message)

    async def take_picture(message):
        """Takes picture and uploads it to channel"""
        capture_and_fetch(username="pi", local_path="./img.jpg")
        with open("./img.jpg", "rb") as f:
            picture = discord.File(f)
            print(f'Sent picture to {message.channel.name} at {message.created_at} triggered by {message.author}')
            await message.channel.send(file=picture)
    
    async def take_and_model_picture(message):
        """Takes picture, runs YOLOv5, and uploads it to channel"""
        capture_and_fetch(username='pi', local_path='./img.jpg')
        model = get_default_model()
        preds, img = predict(model, './img.jpg')
        print(preds)
        await send_file(message, './img_yolo.png')
    
    async def send_file(message, file_path):
        with open(file_path, "rb") as f:
            picture = discord.File(f)
            print(f'Sent picture to {message.channel.name} at {message.created_at} triggered by {message.author}')
            await message.channel.send(file=picture)

    load_dotenv()
    TOKEN = os.getenv("DISCORD_TOKEN")
    client.run(TOKEN)
