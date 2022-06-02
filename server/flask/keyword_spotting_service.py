import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = 'model.h5'
NUM_SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:

    model = None
    mappings = [
        "happy",
        "right",
        "off",
        "wow",
        "up",
        "four",
        "tree",
        "one",
        "six",
        "two",
        "sheila",
        "marvin",
        "down",
        "five",
        "nine",
        "stop",
        "eight",
        "three",
        "yes",
        "bird",
        "house",
        "seven",
        "left",
        "zero",
        "no",
        "on",
        "go",
        "bed",
        "cat",
        "dog"
    ]

    _instance = None

    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path) # segments, coefficients

        # convert 2D MFCCs array to 4D [samples, segments, coefficients, channels]
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # predict
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self.mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):

        # load audio
        signal, sr = librosa.load(file_path)

        # ensure consistency in file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]
        
        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T

def Keyword_Spotting_Service():

    # singleton logic
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

# test class
if __name__ == '__main__':

    kss = Keyword_Spotting_Service()

    keyword1 = kss.predict('test/down.wav')
    keyword2 = kss.predict('test/left.wav')

    print(f"Predicted: {keyword1}, {keyword2}")