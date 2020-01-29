import sys
import traceback
import numpy as np
import support.config as c
from keras.models import load_model
from utils.train_util import *


def main(string):
    assert len(string) == c.str_len, 'string len != ' + str(c.str_len) + \
        ', retrain the model with correspoing string length'
    try:
        model = load_model(c.model_path)
    except FileNotFoundError:
        print('Model not found, try retraining')
    one_hot_class = model.predict(np.array([process_string(string)]))
    one_hot_class = np.around(np.around(one_hot_class[0]))
    for i in range(21):
        if one_hot_class[i] == 1.:
            bucket = i
    print(bucket)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        try:
            main(str(sys.argv[1]))
        except Exception as e:
            print(e)
            traceback.print_exc()
    else:
        print('using dummy string:')
        dummy_string = '10001010100001010000'
        print(dummy_string)
        main(dummy_string)
