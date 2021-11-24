import os
import time

import pandas as pd
import requests as req

from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

DATA_DIR = './nbsr'
DATA_OUT_DIR = './nbsr/data/out'
CHROME_PATH ='helper/chromedriver.exe'

YEAR = 0
MONTH = 1
DAY = 2
STOCK_NAME = '삼성전자'
SLEEP_SEC = 0.3;
SCROLL_PAUSE_SEC = 1;

def parse_url(url, params=None):
    # 브라우저를 통해서 접근하는 것처럼 header 설정
    r = req.get(url, params, headers={'User-Agent': 'Mozilla/5.0'})

    if r.status_code == 200:  # OK
        return BeautifulSoup(r.text, 'html.parser')
    else:
        return None
    
def select_calendar_input_data(wrap_box, index, value):
    type_of_input_box = wrap_box.find_elements_by_xpath('.//div[@class="group_select _list_root"]')[index]
    input_box_tab_list = type_of_input_box.find_element_by_xpath(
        './/div[@class="select_cont"]/div[@class="select_area _scroll_wrapper"]/div/ul[@role="tablist" and @class="lst_item _ul"]')
    time.sleep(SLEEP_SEC)
    
    input_box_tab_elem_list = input_box_tab_list.find_elements_by_xpath('.//li[@class="item _li"]/a[@class="link"]')
    
    target_button = [el for el in input_box_tab_elem_list if int(el.get_attribute('innerHTML')) == value][0]
    target_button.send_keys('\n')


def setting_date_options(browser, year, month, day):
    """## 검색 옵션 설정 과정"""   
    option_group_box = browser.find_element_by_xpath(
        './/div[@class="api_group_option_sort _search_option_detail_wrap"]')
    browser.execute_script('arguments[0].setAttribute("style", "display: block;")', option_group_box)
      
    option_group_list_box = option_group_box.find_element_by_xpath(
        './/ul[@class="lst_option"]')
    time.sleep(SLEEP_SEC)
    
    direct_input_button = option_group_list_box.find_element_by_xpath(
        './/li[@class="bx term"]/div[@class="bx_inner"]/div[@class="option"]/a[@class="txt txt_option _calendar_select_trigger"]')
    direct_input_button.click()
    
    select_calendar_box = option_group_list_box.find_element_by_xpath(
        './/div[@class="api_select_option type_calendar _calendar_select_layer"]')
    time.sleep(SLEEP_SEC)
    select_calendar_input_target = select_calendar_box.find_element_by_xpath(
        './/div[@class="set_calendar"]')
    
    """## 기한 설정(from)"""
    select_calendar_input_wrap = select_calendar_box.find_element_by_xpath(
        './/div[@class="select_wrap _root"]')
    time.sleep(SLEEP_SEC)
    
    select_calendar_input_data(select_calendar_input_wrap, YEAR, year)
    select_calendar_input_data(select_calendar_input_wrap, MONTH, month)
    select_calendar_input_data(select_calendar_input_wrap, DAY, day)
    
    """## 기한 설정(to)"""
    select_calendar_input_target_button = select_calendar_input_target.find_element_by_xpath(
        './/span[@class="set"]/a[@class="spnew_bf ico_calendar _end_trigger"]')
    select_calendar_input_target_button.click()
    
    select_calendar_input_wrap = select_calendar_box.find_element_by_xpath(
        './/div[@class="select_wrap _root"]')
    time.sleep(SLEEP_SEC)
    
    select_calendar_input_data(select_calendar_input_wrap, YEAR, year)
    select_calendar_input_data(select_calendar_input_wrap, MONTH, month)
    select_calendar_input_data(select_calendar_input_wrap, DAY, day)
    
    """## 옵션 적용"""
    apply_button = select_calendar_box.find_element_by_xpath(
        './/div[@class="btn_area"]/button[@class="btn_apply _apply_btn"]')
    apply_button.click()
    
def crawling_main_text(url):
    soup = parse_url(url)
    text = []
    for div in soup.select('p'):
        text.append(div.text.strip().replace('\n', '').replace('\u200b', ''))
    return ' '.join(text)

raw_data = pd.DataFrame(columns=['genre', 'title', 'text'])
print('본문 크롤링에 필요한 함수를 로딩하고 있습니다...\n' + '-' * 100)

chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('headless')
# chrome_options.add_argument("--disable-gpu")
# chrome_options.add_argument("lang=ko_KR")

chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
)

print('브라우저를 실행시킵니다(자동 제어)\n')
browser = webdriver.Chrome("{}/{}".format(DATA_DIR,CHROME_PATH), options=chrome_options)


"""## 쿼리문 생성 및 검색어 입력"""

query = STOCK_NAME
views_url = 'https://search.naver.com/search.naver?where=news&query={}'.format(query)
browser.get(views_url)

setting_date_options(browser, 2020, 10, 23)
time.sleep(2)

setting_date_options(browser, 2020, 8, 23)
time.sleep(3)

browser.close()