import requests
from bs4 import BeautifulSoup
# 取得網頁內容
url = 'https://water.taiwanstat.com/'
web = requests.get(url) 
#寫到文字檔
f = open("return.html", "w",encoding='utf8')
f.write(web.text)
f.close()


soup = BeautifulSoup(web.text, "html.parser")  # 轉換成標籤樹
title = soup.title                             # 取得 title

allnames=soup.select('div.name')
item=allnames[0]

print(item.text)
print(item.find('h3').text)