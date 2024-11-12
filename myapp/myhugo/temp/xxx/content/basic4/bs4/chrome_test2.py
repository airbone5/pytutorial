from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

s=Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s)
driver.maximize_window()
url = 'https://water.taiwanstat.com/'
driver.get(url)

#elements = driver.find_elements(By.XPATH, '//*[@id="userActivityGraph"]')
#Elements is a selenium WebElement, so we will need to get HTML out of it.

#svg = [WebElement.get_attribute('innerHTML') for WebElement in elements]