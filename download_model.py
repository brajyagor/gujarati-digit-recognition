import urllib.request

url = "https://github.com/brajyagor/gujarati-digit-recognition/blob/main/ghDigitReco_10072025_1.h5"
output = "ghDigitReco_10072025_1.h5"

urllib.request.urlretrieve(url, output)

print("Model downloaded from GitHub")
