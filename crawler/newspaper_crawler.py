import csv

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm


def parse_header(raw_article):
    soup = BeautifulSoup(raw_article, "html.parser")

    title = soup.find("h1", class_="article-detail-title f0")
    datetime = soup.find("span", class_="article-detail-publish")

    return str(title.text).strip(), str(datetime.text).strip()


def parse_paragraph(raw_article):
    soup = BeautifulSoup(raw_article, "html.parser")
    paragraphs = soup.find_all("p")

    content = ""
    for paragraph in paragraphs:
        content += paragraph.text

    return str(content.strip()).strip()


def crawl_article_by_numOfPages(num_pages):
    num_pages = num_pages if num_pages <= 236 else 236

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(options=chrome_options)
    delay = 10

    topic_url = "https://vietnamtimes.org.vn/economy"
    fields = ["Title", "Datetime", "Description"]
    for i in tqdm(range(1, num_pages + 1)):
        data = []
        page_url = topic_url + "&s_cond=&BRSR={}".format((i - 1) * 20)
        driver.get(page_url)

        WebDriverWait(driver, delay).until(EC.presence_of_element_located((
            By.CSS_SELECTOR,
            "h3.article-title"
        )))
        articles_wrapper = driver.find_element(By.CSS_SELECTOR, "div.col-Left.lt")
        articles_wrapper_raw = articles_wrapper.get_attribute("outerHTML")
        soup = BeautifulSoup(articles_wrapper_raw, "html.parser")

        url_wrappers = soup.find_all("a", class_="article-image")
        for url_wrapper in url_wrappers:
            article_url = url_wrapper.attrs["href"]
            print(article_url)
            driver.get(article_url)

            try:
                WebDriverWait(driver, delay).until(EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    "div.google-auto-placed.ap_container")
                ))
            except Exception as e:
                print(f"Error loading page {article_url}: {e}")
                continue

            info_wrapper = driver.find_element(By.CSS_SELECTOR, "div.article-detail.fw.clearfix")
            info_wrapper_raw = info_wrapper.get_attribute("outerHTML")
            title, datetime = parse_header(info_wrapper_raw)

            content_wrapper = driver.find_element(By.CSS_SELECTOR, "div#__MB_MASTERCMS_EL_3")
            content_wrapper_raw = content_wrapper.get_attribute("outerHTML")

            content = parse_paragraph(content_wrapper_raw)

            data.append([title, datetime, content])

        with open("data//article.csv", "a", encoding="utf-8") as f:
            write = csv.writer(f)
            write.writerows(data)
            

if __name__ == "__main__":
    num_pages = int(input("Enter the desired number of pages for crawling: "))
    crawl_article_by_numOfPages(num_pages)