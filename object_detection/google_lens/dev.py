from serpapi import GoogleSearch

api_key = input('API:')

params = {
  "engine": "google_lens",
  # "url": "https://i.imgur.com/HBrB8p0.png",
  "url": "https://github.com/ErenBYilmaz/video-captioning/blob/master/resources/ImageMetadataFromVideo_VIDEO_CAPTIONING_DEMO_VIDEO_mp4_00-00-23-000.jpg?raw=true",
  "api_key": api_key
}

search = GoogleSearch(params)
results = search.get_dict()
visual_matches = results["visual_matches"]
