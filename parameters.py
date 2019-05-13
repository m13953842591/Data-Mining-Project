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
DATA_PATH = "data"      # set your path to save data
MODEL_PATH = "model"    # set your path to save model

# define log directory
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
LOGDIR = 'logs/' + TIMESTAMP

# Hyper parameters
BATCH_SIZE = 1000
NUM_EPOCH = 10
HIDDEN_SIZE = 50
USE_DROPOUT = True
DROPOUT_RATE = 0.5
LR = 0.0001      # learning rate
RR = 0.001      # regularization rate
DECAY = 0.5     # decay rate
