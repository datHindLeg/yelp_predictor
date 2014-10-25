import urllib2
from bs4 import BeautifulSoup
import re
from threading import Thread

def check_url_exists(ur):
    request = urllib2.Request(ur)
    try:
        response = urllib2.urlopen(request)
        html = urllib2.urlopen(ur).read()
        soup = BeautifulSoup(html)
        dates = soup.findAll('meta', {"itemprop":"datePublished"})
        if not dates:
            return False
        else:
            return True
    except urllib2.HTTPError:
        return False

def get_urls():
    urls=['http://www.yelp.com/biz/chicos-pizza-san-francisco']
    all_urls = []
    for item in urls:
        if check_url_exists(item + '?sort_by=date_desc') == False:
            print 'WHAT? Restuarant URL doesnt exist apparently'
            continue
        counter = 40
        temp = []
        temp.append(item + '?sort_by=date_desc')
        while True:
            new_url = item + '?start=' + str(counter) + '&sort_by=date_desc'
            if check_url_exists(new_url):
                temp.append(new_url)
            else:
                all_urls.append(temp)
                break
            counter += 40
    return all_urls

def scrape(ur):
    html = urllib2.urlopen(ur).read()
    soup = BeautifulSoup(html)
    title = soup.find('h1',itemprop="name")
    reviews = soup.findAll('p',itemprop='description')
    ratings = soup.findAll('meta', {"itemprop":"ratingValue"})
    dates = soup.findAll('meta', {"itemprop":"datePublished"})
    print '\n'
    print title.text.strip()
    print ratings[0]['content'] + '\n\n' 
    print "-----------------------------------------------------"
    print "*****************************************************"
    print "-----------------------------------------------------"
    print '\n'
    i = 0
    while (i < len(reviews)):
        print ratings[i+1]['content'] + '\n'
        print dates[i]['content'] + '\n'
        print reviews[i].text
        print "-------------------"
        i += 1

def main():
    for items in get_urls():
        for subitem in items:
            scrape(subitem)

if __name__ == "__main__":
    main()

