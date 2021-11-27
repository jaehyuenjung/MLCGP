import os
import re
import time

import pandas as pd
import requests as req

from tqdm import tqdm
from datetime import date
from bs4 import BeautifulSoup
from selenium import webdriver
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

DATA_DIR = './nbsr'
DATA_IN_DIR = './nbsr/data/in'
DATA_OUT_DIR = './nbsr/data/out'
CHROME_PATH ='helper/chromedriver.exe'
FILE_PATH = 'news_data_collection.xlsx'
STOCK_DATA_FILE_PATH = 'stock_ko/samsung_20000101-20210926.csv';

YEAR = 0
MONTH = 1
DAY = 2
CONSONANT = 0
WORD = 1
YONHAP_NEWS = 0
CHOSUN_ILBO = 1
KBS = 2
SBS = 3
KYUNGHYANG_NEWSPAPER = 4
STOCK_NAME = '삼성전자'
SLEEP_SEC = 0.5;

"""## 초성 리스트. 00 ~ 18"""
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
"""## 언론사 리스트. 00 ~ 05"""
PRESS_LIST = ['연합뉴스', '조선일보', 'KBS', 'SBS', '경향신문']

def parse_url(url, params=None):
    """## 브라우저를 통해서 접근하는 것처럼 header 설정"""
    r = req.get(url, params, headers={'User-Agent': 'Mozilla/5.0'})
    if r.status_code == 200:  # OK
        return BeautifulSoup(r.text, 'html.parser')
    else:
        return None
    
def select_type_input_data(wrap_box, index, value):
    """## 데이트 타입 설정"""
    type_of_input_box = wrap_box.find_elements_by_xpath('.//div[@class="group_select _list_root"]')[index]
    input_box_tab_list = type_of_input_box.find_element_by_xpath(
        './/div[@class="select_cont"]/div[@class="select_area _scroll_wrapper"]/div/ul[@role="tablist" and @class="lst_item _ul"]')
    time.sleep(SLEEP_SEC)
    
    """## 해당 데이트 타입에 존재하는 값 불러옴"""
    input_box_tab_elem_list = input_box_tab_list.find_elements_by_xpath('.//li[@class="item _li"]/a[@class="link"]')
    time.sleep(SLEEP_SEC)
    
    """## 목표 값이랑 같은 것을 선택"""
    target_button = [el for el in input_box_tab_elem_list if el.get_attribute('innerHTML') == str(value)][0]
    target_button.send_keys('\n')
    time.sleep(SLEEP_SEC)


def setting_date_options(browser, fy, fm, fd, ty, tm, td):
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
    
    select_type_input_data(select_calendar_input_wrap, YEAR, fy)
    select_type_input_data(select_calendar_input_wrap, MONTH, fm)
    select_type_input_data(select_calendar_input_wrap, DAY, fd)
    
    """## 기한 설정(to)"""
    select_calendar_input_target_button = select_calendar_input_target.find_element_by_xpath(
        './/span[@class="set"]/a[@class="spnew_bf ico_calendar _end_trigger"]')
    select_calendar_input_target_button.click()
    time.sleep(SLEEP_SEC)
    
    select_calendar_input_wrap = select_calendar_box.find_element_by_xpath(
        './/div[@class="select_wrap _root"]')
    time.sleep(SLEEP_SEC)
    
    select_type_input_data(select_calendar_input_wrap, YEAR, ty)
    select_type_input_data(select_calendar_input_wrap, MONTH, tm)
    select_type_input_data(select_calendar_input_wrap, DAY, td)
    
    """## 옵션 적용"""
    apply_button = select_calendar_box.find_element_by_xpath(
        './/div[@class="btn_area"]/button[@class="btn_apply _apply_btn"]')
    apply_button.click()
    time.sleep(SLEEP_SEC)
    
def format_korean_to_consonant(korean_word):
    w = korean_word[0]
    
    if '가'<=w<='힣':
        ## 588개 마다 초성이 바뀜. 
        ch = (ord(w) - ord('가'))//588
        return CHOSUNG_LIST[ch]
    return None

def select_press_input_data(wrap_box, index, value):
    """## 데이트 타입 설정(년, 월, 일)"""
    type_of_input_box = wrap_box.find_elements_by_xpath('.//div[@class="group_select _list_root"]')[index]
    input_box_tab_list = type_of_input_box.find_element_by_xpath(
        './/div[@class="select_cont"]/div[@class="select_area _scroll_wrapper"]/div/ul[@role="tablist" and @class="lst_item _ul"]')
    time.sleep(SLEEP_SEC)
    
    """## 해당 타입에 존재하는 값 불러옴"""
    input_box_tab_elem_list = input_box_tab_list.find_elements_by_xpath('.//li[@class="item _li"]/a[@class="link"]')
    time.sleep(SLEEP_SEC)
    
    """## 목표 값이랑 같은 것을 선택"""
    target_button = [el for el in input_box_tab_elem_list if int(el.get_attribute('innerHTML')) == value][0]
    target_button.send_keys('\n')
    time.sleep(SLEEP_SEC)
    
def setting_press_options(browser, press):
    """## 검색 옵션 설정 과정"""   
    option_group_box = browser.find_element_by_xpath(
        './/div[@class="api_select_option type_dictionary _abc_select_layer"]')
    browser.execute_script('arguments[0].setAttribute("style", "display: block;")', option_group_box)
    
    select_press_input_wrap = option_group_box.find_element_by_xpath(
        './/div[@class="select_wrap _root"]')
    time.sleep(SLEEP_SEC)
    
    """## 언론사 초성을 구한 뒤 한국어면 if 동작 영어면 else 동작"""
    consonant = format_korean_to_consonant(press)
    if consonant:
        select_type_input_data(select_press_input_wrap, CONSONANT, format_korean_to_consonant(press))
    else:
        select_type_input_data(select_press_input_wrap, CONSONANT, 'ABC')
        
    select_type_input_data(select_press_input_wrap, WORD, press)
    
def news_section_crawling(url, press_index):
    """## 불필요한 공백 제거"""
    p = re.compile(r'\s+')
    
    soup = parse_url(url, {'encoding': 'utf-8'})
    
    try: 
        if press_index == YONHAP_NEWS:
            try:
                content_of_article = soup.find('article', {'class':'story-news article'}).find_all('p')
            except:
                try:
                    content_of_article = soup.find('div', {'class':'article-txt'}).find_all('p')
                except:
                    try:
                        return  p.sub(' ', soup.find('div', {'id':'articleBody'}).text).strip()
                    except:
                        return  p.sub(' ', soup.find('div', {'class':'end_body_wrp'}).text).strip()
                        
            string_list = []
            
            for item in content_of_article:
                sent = p.sub(' ', ' '.join(list(map(str, item.find_all(text=True)))))
                
                string_list.append(sent)

            return p.sub(' ', ' '.join(string_list)).strip()
        
        elif press_index == CHOSUN_ILBO:
            new_url = soup.find('link', {'rel':'amphtml'})['href']
            
            soup = parse_url(new_url, {'encoding': 'utf-8'})
            content_of_article = soup.find('section', {'class':'article-body'}).find_all('p')
            
            string_list = []
            
            for item in content_of_article:
                sent = p.sub(' ', ' '.join(list(map(str, item.find_all(text=True)))))
                
                string_list.append(sent)

            return p.sub(' ', ' '.join(string_list)).strip()
        
        elif press_index == KBS:
            content_of_article = soup.select('div.detail-body')
            
            string_list = []
            
            for item in content_of_article:
                sent = p.sub(' ', ' '.join(list(map(str, item.find_all(text=True)))))
                
                string_list.append(sent)
                
            return p.sub(' ', ' '.join(string_list)).strip()
        
        elif press_index == SBS:
            content_of_article = soup.select('div.text_area')
            
            string_list = []
            
            for item in content_of_article:
                sent = p.sub(' ', ' '.join(list(map(str, item.find_all(text=True)))))
                
                string_list.append(sent)
                
            return p.sub(' ', ' '.join(string_list)).strip()

        elif press_index == KYUNGHYANG_NEWSPAPER:
            content_of_article = soup.select('p.content_text') 
            string_list = []
            
            for item in content_of_article:
                sent = p.sub(' ', ' '.join(list(map(str, item.find_all(text=True)))))
                
                """## 주석 제거"""
                sent = re.sub(r"SUB_TITLE_START", "", sent)
                sent = re.sub(r"SUB_TITLE_END", "", sent)
                
                string_list.append(sent)

            return p.sub(' ', ' '.join(string_list)).strip()
    except:
        return None


def cur_page_crawling(url, press):
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
        
        """## 언론사에 해당하는 고유 인덱스로 변환"""
        press_index = PRESS_LIST.index(press)
        
        """## 뉴스 데이터 생성"""
        news_date = info_date.text
        news_url =  tit.get('href')
        news_press = info_press.text
        news_title = tit.get('title')
        news_section = news_section_crawling(news_url, press_index)
        
        if news_section:
            news_data = {'date': news_date, 'url': news_url, 'press': news_press,  'title': news_title, 'section': ILLEGAL_CHARACTERS_RE.sub(r'',news_section)}
            cur_news_data_list.append(news_data)
        
    return cur_news_data_list
        
def set_news_collection_crawling(browser, pr):
    """## 현재 웹 페이지에 존재하는 페이지의 수 추출"""
    page_total_count = len(browser.find_elements_by_xpath(
        './/div[@class="api_sc_page_wrap"]/div[@class="sc_page"]/div[@class="sc_page_inner"]/a[@role="button" and @class="btn"]'))
    time.sleep(1)
    
    news_data_list = []
    try:
        for n in range(page_total_count):
            """## 해당 페이지 버튼 클릭"""
            page_button = browser.find_elements_by_xpath(
            './/div[@class="api_sc_page_wrap"]/div[@class="sc_page"]/div[@class="sc_page_inner"]/a[@role="button" and @class="btn"]')[n]
            page_button.click()
            time.sleep(1)
            
            cur_url = browser.current_url;
            cur_news_data_list = cur_page_crawling(cur_url, pr)
            news_data_list.extend(cur_news_data_list)
    
    finally: return news_data_list

def make_big_news_data_set(browser, start, end):
    """## 목표 파일이 존재하면 불러오고 없다면 새로 만듬"""
    try:
        raw_data = pd.read_excel('{}/{}'.format(DATA_OUT_DIR, FILE_PATH))
    except:
        raw_data = pd.DataFrame(columns=['date', 'url', 'press', 'title', 'section'])
        
    pbar = tqdm(total=len(PRESS_LIST))
    
    setting_date_options(browser, start.year, start.month, start.day, end.year, end.month, end.day)

    for pr in PRESS_LIST:
        
        setting_press_options(browser, pr)
        news_data_list = set_news_collection_crawling(browser, pr)

        """## 데이터 저장"""
        if news_data_list:
            for news_data in news_data_list:
                raw_data = raw_data.append(news_data, ignore_index=True)
        pbar.update(1)
        time.sleep(SLEEP_SEC)      
        
    pbar.close()
    
    """## 최종 파일을 엑셀 형식으로 저장"""
    print('\n전체 크롤링한 결과를 데이터프레임 형식으로 저장 중')
    raw_data.to_excel('{}/{}'.format(DATA_OUT_DIR, FILE_PATH), index=False, encoding='utf-8')
    print('\n저장을 완료하였습니다.\n')
    
def format_date_to_obj(dt):
    year, month, day = dt.split('/');
    return date(int(year), int(month), int(day))
    
def make_mini_batch_data_set(browser, df):
    date_list = df['일자'].values.tolist()
    date_list = list(map(format_date_to_obj, date_list))
    index = 0
    list_size = len(date_list)
    
    while index < list_size:
        pre_date = date_list[index]
        """## 최대 연속적인 날짜 5개까지 데이터(월, 화, 수, 목, 금)"""
        mini_batch_list = [date_list[index]]
        
        """## 처음 데이터 포함 최대 4개 데이터를 저장"""
        for n in range(4):
            index += 1
            
            if(index >= list_size):
                break
            day_diff = (pre_date - date_list[index]).days
            """## 다음 날짜 데이터가 하루단위로 연속적이라면"""
            if day_diff == 1:
                pre_date = date_list[index]
                mini_batch_list.append(date_list[index])
                
                if n == 3:
                    index += 1
            else:
                break
        mini_batch_list.reverse()
        make_big_news_data_set(browser, mini_batch_list[0], mini_batch_list[-1]) 
  
df = pd.read_csv('{}/{}'.format(DATA_IN_DIR, STOCK_DATA_FILE_PATH))[1727 : 3455]
    
print('본문 크롤링에 필요한 함수를 로딩하고 있습니다...\n' + '-' * 100)

"""## 크롬 드라이버 옵션 설정"""
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('headless')
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("lang=ko_KR")
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
)

print('브라우저를 실행시킵니다(자동 제어)\n')
browser = webdriver.Chrome("{}/{}".format(DATA_DIR,CHROME_PATH), options=chrome_options)

"""## 쿼리문 생성 및 검색어 입력"""
query = STOCK_NAME
views_url = 'https://search.naver.com/search.naver?where=news&query={}'.format(query)
browser.get(views_url)

print('\n크롤링을 시작합니다.')
time.sleep(SLEEP_SEC)

make_mini_batch_data_set(browser, df)
time.sleep(SLEEP_SEC)

browser.close()

print('뉴스 기사 크롤링을 완료하였습니다.\n' + '-' * 100)
