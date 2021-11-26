import os
import re
import time

import pandas as pd
import requests as req

from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

DATA_DIR = './nbsr'
DATA_OUT_DIR = './nbsr/data/out'
FILE_PATH = 'news_data_collection.xlsx'
CHROME_PATH ='helper/chromedriver.exe'

YEAR = 0
MONTH = 1
DAY = 2
STOCK_NAME = '삼성전자'
SLEEP_SEC = 0.3;
SCROLL_PAUSE_SEC = 1;

def parse_url(url, params=None):
    """## 브라우저를 통해서 접근하는 것처럼 header 설정"""
    r = req.get(url, params, headers={'User-Agent': 'Mozilla/5.0'})
    r.encoding = 'utf-8' 
    if r.status_code == 200:  # OK
        return BeautifulSoup(r.text, 'html.parser')
    else:
        return None
    
def select_calendar_input_data(wrap_box, index, value):
    """## 데이트 타입 설정(년, 월, 일)"""
    type_of_input_box = wrap_box.find_elements_by_xpath('.//div[@class="group_select _list_root"]')[index]
    input_box_tab_list = type_of_input_box.find_element_by_xpath(
        './/div[@class="select_cont"]/div[@class="select_area _scroll_wrapper"]/div/ul[@role="tablist" and @class="lst_item _ul"]')
    time.sleep(SLEEP_SEC)
    
    """## 해당 데이트 타입에 존재하는 값 불러옴"""
    input_box_tab_elem_list = input_box_tab_list.find_elements_by_xpath('.//li[@class="item _li"]/a[@class="link"]')
    time.sleep(SLEEP_SEC)
    
    """## 목표 값이랑 같은 것을 선택"""
    target_button = [el for el in input_box_tab_elem_list if int(el.get_attribute('innerHTML')) == value][0]
    target_button.send_keys('\n')
    time.sleep(SLEEP_SEC)


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
    select_calendar_input_target = select_calendar_box.find_element_by_xpath(
        './/div[@class="set_calendar"]')
    time.sleep(SLEEP_SEC)
    
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
    time.sleep(SLEEP_SEC)
    
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
    time.sleep(SLEEP_SEC)
    
def news_section_crawling(url):
    # TODO: 언론사에 맞게 크롤링, 인코딩 처리
    soup = parse_url(url)
    text = []
    for div in soup.select('p'):
        text.append(div.text.strip().replace('\n', '').replace('\u200b', ''))
    return ' '.join(text)

def cur_page_crawling(url):
    soup = parse_url(url)
    
    table = soup.find('ul',{'class' : 'list_news'})
    li_list = table.find_all('li', {'id': re.compile('sp_nws.*')})
    area_list = [li.find('div', {'class' : 'news_area'}) for li in li_list]
    
    """## 뉴스 데이터에 필요한 태그 추출"""
    info_list = [area.find('div', {'class' : 'news_info'}) for area in area_list]
    a_list = [area.find('a', {'class' : 'news_tit'}) for area in area_list]
    
    cur_news_data_list = []
    for tit, info in zip(a_list, info_list):
        info_group = info.find('div', {'class' : 'info_group'})
        info_press = info_group.find('a', {'class' : 'info press'})
        info_date = info_group.find_all('span', {'class' : 'info'})[-1]
        
        """## 뉴스 데이터 생성"""
        news_date = info_date.text
        news_url =  tit.get('href')
        news_press = info_press.text
        news_title = tit.get('title')
        news_section = news_section_crawling(news_url)
        
        news_data = {'date': news_date, 'url': news_url, 'press': news_press,  'title': news_title, 'section': news_section}
        cur_news_data_list.append(news_data)
        
    return cur_news_data_list
        
def set_news_collection_crawling(browser):
    """## 현재 웹 페이지에 존재하는 페이지의 수 추출"""
    page_total_count = len(browser.find_elements_by_xpath(
        './/div[@class="api_sc_page_wrap"]/div[@class="sc_page"]/div[@class="sc_page_inner"]/a[@role="button" and @class="btn"]'))
    time.sleep(SLEEP_SEC)
    
    news_data_list = []
    for n in range(page_total_count):
        """## 해당 페이지 버튼 클릭"""
        page_button = browser.find_elements_by_xpath(
        './/div[@class="api_sc_page_wrap"]/div[@class="sc_page"]/div[@class="sc_page_inner"]/a[@role="button" and @class="btn"]')[n]
        page_button.click()
        time.sleep(SLEEP_SEC)
        
        cur_url = browser.current_url;
        cur_news_data_list = cur_page_crawling(cur_url)
        news_data_list.extend(cur_news_data_list)
        
    return news_data_list

"""## 목표 파일이 존재하면 불러오고 없다면 새로 만듬"""
try:
    raw_data = pd.read_excel('{}/{}'.format(DATA_OUT_DIR, FILE_PATH))
except:
    raw_data = pd.DataFrame(columns=['date', 'url', 'press', 'title', 'section'])
    
print('본문 크롤링에 필요한 함수를 로딩하고 있습니다...\n' + '-' * 100)

"""## 크롬 드라이버 옵션 설정"""
chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('headless')
# chrome_options.add_argument("--disable-gpu")
# chrome_options.add_argument("lang=ko_KR")
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
)

print('브라우저를 실행시킵니다(자동 제어)\n')
browser = webdriver.Chrome("{}/{}".format(DATA_DIR,CHROME_PATH), options=chrome_options)

"""## 쿼리문 생성 및 검색어 입력"""
query = STOCK_NAME + ' 주가'
views_url = 'https://search.naver.com/search.naver?where=news&query={}'.format(query)
browser.get(views_url)

setting_date_options(browser, 2020, 10, 23)
time.sleep(SLEEP_SEC)

news_data_list = set_news_collection_crawling(browser)

browser.close()

print('뉴스 기사 크롤링을 완료하였습니다.')
time.sleep(SLEEP_SEC)

"""## 데이터 저장"""
for news_data in news_data_list:
        raw_data = raw_data.append(news_data, ignore_index=True)

"""## 최종 파일을 엑셀 형식으로 저장"""
print('\n전체 크롤링한 결과를 데이터프레임 형식으로 저장 중')
raw_data.to_excel('{}/{}'.format(DATA_OUT_DIR, FILE_PATH), index=False, encoding='utf-8')
print('\n저장을 완료하였습니다.\n' + '-' * 100)