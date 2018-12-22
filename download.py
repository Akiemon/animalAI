from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

#APIキーの情報

key = "9e858bca1bc983e7d50d17904fd73c9e"
secret = "abace57f5ed8cc7d"
wait_time = 1

#保存フォルダの指定
animalname = sys.argv[1]
savedir = "./" + animalname

flickr = FlickrAPI(key, secret, format = "parsed-json")
result = flickr.photos.search(
    text = animalname,
    per_page = 400,
    media = "photos",
    sort = "relevance",
    safe_search = 1,
    extras = "url_q, licence"
)

photos = result["photos"]
# pprint(photos)

for i, photo in enumerate(photos["photo"]):
    url_q = photo["url_q"]
    filepath = savedir + "/" + photo["id"] + ".jpg"
    # 重複を防止する
    if os.path.exists(filepath):
        continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)


