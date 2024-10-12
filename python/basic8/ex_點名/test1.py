
#to access BSD socket interface
import socket

#to work with network servers
import socketserver

#to display the documents to user on other device as a web page
import webbrowser

#to generate a qr code
import pyqrcode
from pyqrcode import QRCode


#to access all directories and os
import os

#assign port
PORT = 8010
hostname = socket.gethostname()

 
#to find the IP address of pc
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #如果類比開檔案,這裡是指定名稱
s.connect(("8.8.8.8", 80))  #如果類比開檔案,這裡才是開檔案 open

IP = "http://" + s.getsockname()[0]+ ":" + str(PORT)
link = IP
 

#convert the IP address into a Qrcode using pyqrcode

url = pyqrcode.create(link)
url.svg("myqr.svg", scale=8)
webbrowser.open('myqr.svg')
# 在终端打印二维码
#print(qr.terminal(quiet_zone=1))
