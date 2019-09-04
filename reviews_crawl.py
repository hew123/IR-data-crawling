import bs4
import urllib.parse
import array as arr 
from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as  soup

urlList = [
    "https://www.yelp.com/biz/sakagura-new-york",
    "https://www.yelp.com/biz/sakagura-new-york?start=20",
    "https://www.yelp.com/biz/sakagura-new-york?start=40",
    "https://www.yelp.com/biz/sakagura-new-york?start=60",
    "https://www.yelp.com/biz/sakagura-new-york?start=80",
    "https://www.yelp.com/biz/sakagura-new-york?start=100",
	"https://www.yelp.com/biz/sakagura-new-york?start=120"
    ]

filename = "products.csv"
f = open(filename,"w")
headers = "ID,RestaurantID,Reviews"
f.write(headers+ "\n")

j=0	
while j < len(urlList):

    myurl = urlList[j]
    uClient = ureq(myurl)
    page = uClient.read()
    uClient.close()
	
    #html.parsing
    page_soup = soup(page,"html.parser")
    #grabs each product
    containers = page_soup.findAll("div",{"class":"review-content"})
    i=1
    while i < len(containers): 
        reviews = containers[i].p.text
        rating = containers[i].div.div.find('div')['title']
        print(i)
        print(reviews)
        print(rating)
        f.write(str(i) +","+"60"+"," +reviews.replace(",",";")+","+rating+ "\n")   
        i+=1
		
	
    j+=1

f.close()