import requests
from bs4 import BeautifulSoup
# 取得網頁內容
url = 'https://water.taiwanstat.com/'
web = requests.get(url) 
soup = BeautifulSoup(web.text, "html.parser")  # 轉換成標籤樹


allnames=soup.select('div.reservoir')
for item in allnames:
    print(item.find('h3').text)
    print(item.find('h5').text)
    print('\n')
