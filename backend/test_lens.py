import requests
def test():
    url = "https://lens.google.com/v3/upload"
    image_data = b"dummy image data"
    files = {'encoded_image': ('upload.jpg', image_data, 'image/jpeg')}
    try:
        response = requests.post(url, files=files, timeout=10, allow_redirects=True)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res2 = requests.get(response.url, headers=headers)
        with open("lens_output.html", "w", encoding="utf-8") as f:
            f.write(res2.text)
        print("Wrote lens_output.html")
    except Exception as e:
        print("Error:", e)
test()
