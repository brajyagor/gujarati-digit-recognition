import urllib.request

url = "https://media.githubusercontent.com/media/brajyagor/gujarati-digit-recognition/main/ghDigitReco_10072025_1.h5"
output = "ghDigitReco_10072025_1.h5"

urllib.request.urlretrieve(url, output)

print("Model downloaded from GitHub")
