

if __name__ == "__main__":
    import datetime
    from utils.train_util import *

    print('Code Started at ' + datetime.datetime.fromtimestamp(time.time()
                                                               ).strftime('%Y-%m-%d %H:%M:%S'))
    model_training()

    print('Code Endeded at ' + datetime.datetime.fromtimestamp(time.time()
                                                               ).strftime('%Y-%m-%d %H:%M:%S'))
