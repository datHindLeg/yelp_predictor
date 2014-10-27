#!/usr/bin/env python
# encoding: utf-8

import urllib2
from bs4 import BeautifulSoup
import re
from threading import Thread
import csv
import time
import datetime

def check_url_exists(ur):
    try:
        urllib2.urlopen(ur)
        return True
    except urllib2.HTTPError:
        return False
    except urllib2.URLError:
        return False

# returns a list of lists, each sublist being a page of reviews of a restaurant
def get_urls():
    # below 5 pages of reviews, standard
    #urls=['http://www.yelp.com/biz/chicos-pizza-san-francisco']
    # below url has no yelp inspections on site
    #urls=['http://www.yelp.com/biz/ocean-pearl-restaurant-san-francisco']
    #urls=['http://www.yelp.com/biz/quickly-san-francisco-12']
    #below url has 100 score
    #urls=['http://www.yelp.com/biz/state-bird-provisions-san-francisco']

    #below is mix of 100 and lesser scores
    urls=['http://www.yelp.com/biz/lite-bite-san-francisco']
    all_urls = []
    for item in urls:
        if check_url_exists(item + '?sort_by=date_desc') == False:
            print 'WHAT? Restuarant URL doesnt exist apparently'
            continue
        counter = 40
        iter = 1
        temp = []
        temp.append(item + '?sort_by=date_desc')
        html = urllib2.urlopen(item + '?sort_by=date_desc').read()
        soup = BeautifulSoup(html)
        page = soup.find('div', {"class":"page-of-pages"})
        while iter < int(page.text.split()[-1]):
            new_url = item + '?start=' + str(counter) + '&sort_by=date_desc'
            if check_url_exists(new_url):
                temp.append(new_url)
            counter += 40
            iter += 1
        all_urls.append(temp)
    print 'Master list ready...'
    return all_urls

# takes a url and scrapes attributes we want from the html, writes to a csv
def scrape(ur, filer, iattrib):
    html = urllib2.urlopen(ur).read()
    soup = BeautifulSoup(html)
    title = soup.find('h1',itemprop="name")
    reviews = soup.findAll('p',itemprop='description')
    ratings = soup.findAll('meta', {"itemprop":"ratingValue"})
    dates = soup.findAll('meta', {"itemprop":"datePublished"})
    passport_stats = soup.findAll('ul', {"class":"user-passport-stats"})

    category = soup.find('meta',{"property":"og:description"})['content'].encode("utf-8").strip(' \t\n\r')

    price_category = soup.find('dd',{"class":"nowrap price-description"}).text.encode('utf-8').strip(' \t\n\r')

    #for yelp inspection section


    name = title.text.encode("utf-8").strip(' \t\n\r')
    total_rating = float( ratings[0]['content'] )
    i = 0
    while (i < len(reviews)):
        if passport_stats[i].find('li', {"class":"is-elite"}) == None:
            elite_status = 'No'
        else:
            elite_status = 'Yes'
        rating = float( ratings[i+1]['content'] )
        date = dates[i]['content'] 
        review_text = reviews[i].text.encode("utf-8").strip(' \t\n\r')
        filer.writerow([name,total_rating,category,price_category,elite_status,rating,date,review_text] + iattrib)
        i += 1

# scrapes the yelp inspection page for each restaurant, returns a list of attributes for scrape() to write to csv
# the reason this isn't performed in scrape() is so a GET request isn't sent for every suburl, only once per restaurant
def scrape_inspection(ur):
    temp = []
    html = urllib2.urlopen(ur).read()
    soup = BeautifulSoup(html)
    recent_inspec_score = soup.find('span',{"class":"score"})
    checker = soup.findAll('td', {"class":"violations text-center"})
    header = soup.find('p', {"class":"catcher"})
    
    recent_inspec = header.text.split('—'.decode('utf-8'))[0].strip(' \t\n\r')
    recent_inspec_type = header.text.split('—'.decode('utf-8'))[1].strip(' \t\n\r')
    number_inspections = 0
   
    if checker != None:
        number_inspections = len(checker) + 1
    else:
        if header != None:
            number_inspections = 1

    factors = time.strptime(recent_inspec.replace(",",""), "%B %d %Y")
    recent_inspec_rd = str(datetime.datetime(factors[0], factors[1], factors[2])).split()[0]

    first_inspec_vio_count = 0
    vio = soup.find("div",{"class":"column column-alpha "}).find('ul', {"class":"bullet-list-square violations-list"})

    recent_inspec_vio = ''
    if soup.findAll("p")[1].text.encode('utf-8').strip(' \t\n\r') == "This inspection has no violations.":
        recent_inspec_vio = 'This inspection has no violations.'
    else:
        lister = vio.findAll('li')
        for violation in lister:
            first_inspec_vio_count += 1
            recent_inspec_vio = recent_inspec_vio + '|' + violation.text.encode("utf-8").strip(' \t\n\r')

    temp.append(number_inspections)

    temp.append(int(recent_inspec_score.text))
    temp.append(first_inspec_vio_count)
    temp.append(recent_inspec_rd)
    temp.append(recent_inspec_type)
    temp.append(recent_inspec_vio)

    k = 0
    table = soup.find("table", {"id":"inspections-table"})
    date_wrapper = table.findAll("td",{"class":"violations text-center"})
    score_wrapper = table.findAll("td",{"class":"text-center"})
    bodies = table.findAll("tr")
    while k < number_inspections - 1:
        score = int( bodies[k+1].findAll("td")[4].text.encode("utf-8").strip(' \t\n\r') )

        inspec_type = bodies[k+1].findAll("td")[1].text.encode("utf-8").strip(' \t\n\r')

        #datea = date_wrapper[k].find("b").text.encode('utf-8').strip(' \t\n\r')
        datea = bodies[k+1].findAll("td")[0].text.encode("utf-8").strip(' \t\n\r')
        dateb = time.strptime(datea.replace(",",""), "%B %d %Y")
        datec = str(datetime.datetime(dateb[0], dateb[1], dateb[2])).split()[0]

        vio_count = 0
        rata = date_wrapper[k].find("span",{"class":"violations-count"})
        if rata != None:
            vio_count = int( date_wrapper[k].find("span",{"class":"violations-count"}).text )

        viol = ''
        if  vio_count == 0:
            viol = 'This inspection has no violations.'
        else:
            lister = bodies[k+1].find("ul",{"class":"bullet-list-square violations-list"})
            for violation in lister.findAll("li"):
                viol = viol + '|' + violation.text.encode("utf-8").strip(' \t\n\r')
       

        temp.append(score)
        temp.append(vio_count)
        temp.append(datec)
        temp.append(inspec_type)
        temp.append(viol)
        k += 1

    return temp

# iteratres through a list of urls, and deeper into suburls, each suburl is a page of reviews
def main():
    f = csv.writer(open("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_yelp.csv", "w"))
    f.writerow(["name","total_rating","category","price_category","elite_status","rating","date","review_text", 
                "number_inspections",
                "recent1_inspec_score","recent1_number_violations","recent1_inspec_date","recent1_inspec_type","recent1_inspec_vio",
                "recent2_inspec_score","recent2_number_violations","recent2_inspec_date","recent2_inspec_type","recent2_inspec_vio",
                "recent3_inspec_score","recent3_number_violations","recent3_inspec_date","recent3_inspec_type","recent3_inspec_vio",
                "recent4_inspec_score","recent4_number_violations","recent4_inspec_date","recent4_inspec_type","recent4_inspec_vio",
                "recent5_inspec_score","recent5_number_violations","recent5_inspec_date","recent5_inspec_type","recent5_inspec_vio",
                "recent6_inspec_score","recent6_number_violations","recent6_inspec_date","recent6_inspec_type","recent6_inspec_vio",
                "recent7_inspec_score","recent7_number_violations","recent7_inspec_date","recent7_inspec_type","recent7_inspec_vio",
                "recent8_inspec_score","recent8_number_violations","recent8_inspec_date","recent8_inspec_type","recent8_inspec_vio",
                "recent9_inspec_score","recent9_number_violations","recent9_inspec_date","recent9_inspec_type","recent9_inspec_vio",
                "recent10_inspec_score","recent10_number_violations","recent10_inspec_date","recent10_inspec_type","recent10_inspec_vio",
                "recent11_inspec_score","recent11_number_violations","recent11_inspec_date","recent11_inspec_type","recent11_inspec_vio",
                ])
    for items in get_urls():
        inter = items[0].split('?', 1)[0] 
        inspection_url = inter.replace('biz', 'inspections')
        if check_url_exists(inspection_url) == True:
            inspectors = scrape_inspection(inspection_url)
        else:
            # if there is no health ratings on yelp, leave blank
            inspectors = ['','','','','']
        for subitem in items:
            scrape(subitem, f, inspectors)

if __name__ == "__main__":
    main()

