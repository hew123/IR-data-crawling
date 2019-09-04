import bs4
import urllib.parse
from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as  soup

myurl = "https://www.yelp.com/search?cflt=restaurants&find_loc=New%20York%2C%20NY&start=60"
uClient = ureq(myurl)
page = uClient.read()
uClient.close()

filename = "restaurants1.csv"
f = open(filename,"w")
headers = "ID,Restaurant,Price,Number"
f.write(headers+ "\n")
#html.parsing
page_soup = soup(page,"html.parser")

#grabs each product
containers = page_soup.findAll("div",{"class":"lemon--div__373c0__1mboc mainAttributes__373c0__1r0QA arrange-unit__373c0__1piwO arrange-unit-fill__373c0__17z0h border-color--default__373c0__2oFDT"})

i=1

while i < len(containers): 
    name= containers[i].div.a.text
    contain = containers[i].div.findAll("span",{"class":"lemon--span__373c0__3997G text__373c0__2pB8f priceRange__373c0__2DY87 text-color--normal__373c0__K_MKN text-align--left__373c0__2pnx_ text-bullet--after__373c0__1ZHaA"})
    price = contain[0].text
    rating = containers[i].findAll("div",{"class":"lemon--div__373c0__1mboc attribute__373c0__1hPI_ display--inline-block__373c0__2de_K u-space-r1 border-color--default__373c0__2oFDT"})
    number = rating[0].find('div')['aria-label']
    
    print(i)
    print(name)
    print(price)
    print(number)
    k = i+30
    f.write(str(k) + "," +name + "," +price + "," + number + "\n")    
    i += 1
	
	
f.close()
	
	
  