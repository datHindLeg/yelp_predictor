import urllib
from bs4 import BeautifulSoup
import re
from threading import Thread

#List of yelp urls to scrape
url=['http://www.yelp.com/biz/the-cheese-steak-shop-san-francisco']

i=0
#function that will do actual scraping job
def scrape(ur):
    html = urllib.urlopen(ur).read()
    soup = BeautifulSoup(html)
    title = soup.find('h1',itemprop="name")
    reviews = soup.findAll('p',itemprop='description')
    print title.text
    for review in reviews:
        print review.text
        print "-------------------"

def main():
    scrape(url[0])

if __name__ == "__main__":
    main()

"""
threadlist = []
#making threads
while i<len(url):
          t = Thread(target=scrape,args=(url[i],))
          t.start()
          threadlist.append(t)
          i=i+1

for b in threadlist:
          b.join()
"""
