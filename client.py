import requests

URL = 'http://54.91.78.26/predict'
TEST_AUDIO_FILE_PATH = 'test/left.wav'

if __name__ == '__main__':

    audio_file = open(TEST_AUDIO_FILE_PATH, 'rb')
    values = {'file': (TEST_AUDIO_FILE_PATH, audio_file, 'audio/wav')}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted keyword is: {data['keyword']}")