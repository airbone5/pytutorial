#Simple assignment
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from time import sleep

driver = Chrome()
  
url = 'https://water.taiwanstat.com/'
driver.get(url)
#driver.implicitly_wait(10) # seconds
sleep(5)
elements=driver.find_elements(By.CSS_SELECTOR,'div.reservoir>svg')  

#列出內容
# for item in elements:    
#     print(item.get_attribute('innerHTML'))
# 列出標籤
# for item in elements:
#     print(item.tag_name) 

for item in elements:       
    print(item.find_element(By.CSS_SELECTOR,'text').text) #text.liquidFillGaugeText') )
    #print(item.find_element(By.XPATH, './/text').text)

