import requests
from bs4 import BeautifulSoup
# 取得網頁內容
url = 'https://water.taiwanstat.com/'
web = requests.get(url) 
soup = BeautifulSoup(web.content, "html.parser")  # 轉換成標籤樹


allnames=soup.select('svg')
item=allnames[1]
print(item)
