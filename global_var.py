from datetime import datetime
# For data management
FUTURE_N = 10       # should rebuild dataset if changed

# Path
DATA_PATH = "D:\\zchen\\Documents\\dataset\\Stock_Price_Prediction"

# define log directory
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
LOGDIR = 'logs/' + TIMESTAMP

# train
BATCH_SIZE = 100
SEQUENCE_LEN = 20
STEP_PER_EPOCH = 512
EPOCH = 20
TEST_BATCH = 100
