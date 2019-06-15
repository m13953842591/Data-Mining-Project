from datetime import datetime
# For data management
TIME_STEP = 20
NUM_LABEL = 1
SKIP_SIZE = 5
FUTURE_N = 10       # should rerun data_management if changed
TEST_SIZE = 0.25    # should rerun data_management if changed
FILE_SIZE = 100000  # should rerun data_management if changed
NUM_FEATURE = 137

# Path
DATA_PATH = "D:\\zchen\\Documents\\dataset\\Stock_Price_Prediction"
# set your path to save data

# define log directory
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
LOGDIR = 'logs/' + TIMESTAMP

# train
BATCH_SIZE = 100
