import tensorflow as tf

# define the paths of the datasets
# rppa 
tf.flags.DEFINE_string("rppa", "datasets/rppa", "")

# general parameters 
tf.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
tf.flags.DEFINE_float('dropout_prob', 0., 'Dropout rate')
tf.flags.DEFINE_integer('gcn_layers', 2, 'num of gcn layers')
tf.flags.DEFINE_integer('time_steps', 5, 'time point to predict')
tf.flags.DEFINE_float('train_ratio', 0.7, 'time point to predict')
tf.flags.DEFINE_integer('hidden_dim', 5, 'hidden embed size of gcn')
tf.flags.DEFINE_integer('hidden_size', 6, 'LSTM hidden size')
tf.flags.DEFINE_integer('window_size', 2, 'LSTM prediction window size')
tf.flags.DEFINE_integer('batch_size', 200, 'LSTM training batch size')
tf.flags.DEFINE_integer('epochs', 300, 'Number of epochs to train')