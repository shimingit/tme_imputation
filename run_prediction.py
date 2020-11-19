import time
import tensorflow as tf
import numpy as np
from models import gcn, lstm
from configs import *
from utils import *
import scipy.sparse

np.random.seed(123)
FLAGS = tf.flags.FLAGS

dataset = FLAGS.rppa
time_steps = FLAGS.time_steps
train_ratio = FLAGS.train_ratio

adjs, feats, train_idx, val_idx, test_idx = load_data(dataset, time_steps, train_ratio)

num_node = adjs[0].shape[0]
num_feat = feats[0].shape[1]

for i in range(time_steps):
    adjs[i] = sparse_to_tuple(scipy.sparse.coo_matrix(adjs[i]))
#     feats[i] = sparse_to_tuple(scipy.sparse.coo_matrix(feats[i]))
num_features_nonzeros = [x[1].shape for x in feats]

# define placeholders of the input data 
phs = {
        'adjs': [tf.sparse_placeholder(tf.float32, shape=(None, None), name="adjs") for i in
             range(time_steps)],
        'feats': [tf.placeholder(tf.float32, shape=(None, num_feat), name="feats") for _ in
                 range(time_steps)],
        'train_idx': tf.placeholder(tf.int32, shape=(None,), name="train_idx"),
        'val_idx': tf.placeholder(tf.int32, shape=(None,), name="val_idx"),
        'test_idx': tf.placeholder(tf.int32, shape=(None,), name="test_idx"),
        'sample_idx': tf.placeholder(tf.int32, shape=(FLAGS.batch_size,), name='batch_sample_idx'),
        'dropout_prob': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzeros': [tf.placeholder(tf.int64) for i in range(time_steps)]
        }

# define the GCN model
gcn_model = gcn.GraphConvLayer(time_steps = time_steps,
                               gcn_layers=FLAGS.gcn_layers,
                               input_dim=num_feat,
                               hidden_dim=FLAGS.hidden_dim,
                               output_dim=FLAGS.hidden_size,
                               name='nn_fc1',
                               num_features_nonzeros=phs['num_features_nonzeros'],
                               act=tf.nn.relu,
                               dropout_prob=phs['dropout_prob'],
                               dropout=True)
embeds_list = gcn_model(adjs=phs['adjs'],
                    feats=phs['feats'],
                    sparse=False)

# prepare train data for the LSTM-based prediction model
## replace all missing features at (time_steps-1) with GCN imputed features
# embeds_list[time_steps-1] = tf.add(phs['feats'][time_steps-1], 
#                                    tf.multiply(phs['test_mask'][time_steps-1], embeds_list[time_steps-1]))
## construct training samples for the prediction task
x_train, y_train, x_val, y_val, x_test, y_test = build_train_samples_imputation(embeds_list=embeds_list, 
                                                                     feats=phs['feats'], 
                                                                     train_idx=phs['train_idx'],
                                                                     val_idx=phs['val_idx'],
                                                                     test_idx=phs['test_idx'],
                                                                     time_steps=time_steps)
# define the bi-directional LSTM model
lstm_model = lstm.BiLSTM(hidden_size=FLAGS.hidden_size,
                         seq_len=FLAGS.time_steps-1,
                         holders=phs)
x_input_seq = tf.gather(x_train, phs['sample_idx'])
y_input_seq_real = tf.gather(y_train, phs['sample_idx'])
y_input_seq_pred = lstm_model(input_seq=x_input_seq)

with tf.name_scope('optimizer'):
    # calculate the train mse and ad
    train_mse = tf.losses.mean_squared_error(y_input_seq_real, y_input_seq_pred)
    train_absolute_diff = tf.losses.absolute_difference(y_input_seq_real, y_input_seq_pred)
    
    # calculate the val mse and ad
    val_input_seq_pred = lstm_model(input_seq=x_val)
    val_mse = tf.losses.mean_squared_error(y_val, val_input_seq_pred)
    val_absolute_diff = tf.losses.absolute_difference(y_val, val_input_seq_pred)
    
    # calculate the test mse and ad
    test_input_seq_pred = lstm_model(input_seq=x_test)
    test_mse = tf.losses.mean_squared_error(y_test, test_input_seq_pred)
    test_absolute_diff = tf.losses.absolute_difference(y_test, test_input_seq_pred)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    opt_op = optimizer.minimize(train_mse)

n_cpus = 8
config = tf.ConfigProto(device_count={ "CPU": n_cpus},
                            inter_op_parallelism_threads=n_cpus,
                            intra_op_parallelism_threads=2)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

feed_dict = {phs['train_idx']: train_idx,
             phs['val_idx']: val_idx,
             phs['test_idx']: test_idx,
             phs['sample_idx']: None,
             phs['dropout_prob']: FLAGS.dropout_prob}

feed_dict.update({phs['adjs'][t]: adjs[t] for t in range(time_steps)})
feed_dict.update({phs['feats'][t]: feats[t] for t in range(time_steps)})
feed_dict.update({phs['num_features_nonzeros'][t]: num_features_nonzeros[t] for t in range(time_steps)})

feed_dict_val = {phs['train_idx']: train_idx,
                 phs['val_idx']: val_idx,
                 phs['test_idx']: test_idx,
                 phs['dropout_prob']: 0}

feed_dict_val.update({phs['adjs'][t]: adjs[t] for t in range(time_steps)})
feed_dict_val.update({phs['feats'][t]: feats[t] for t in range(time_steps)})
feed_dict_val.update({phs['num_features_nonzeros'][t]: num_features_nonzeros[t] for t in range(time_steps)})



def get_batch_idx(epoch):
    s = FLAGS.batch_size * epoch
    e = FLAGS.batch_size * (epoch + 1)
    idx = []
    for i in range(s,e):
        idx.append(i%len(train_idx))
    return idx

epochs = FLAGS.epochs
save_step = 10
t = time.time()
for epoch in range(epochs):
    batch_samples = get_batch_idx(epoch)
    feed_dict.update({phs['sample_idx']: batch_samples})
    _, train_MSE, train_AD = sess.run((opt_op, train_mse, train_absolute_diff), feed_dict=feed_dict)
    val_MSE, val_AD = sess.run((val_mse, val_absolute_diff), 
                                         feed_dict=feed_dict_val) 
    
    print("Epoch:", '%04d' % (epoch + 1),
      "train_loss=", "{:.5f}".format(train_MSE),
      "train_MSE=", "{:.5f}".format(train_MSE),
      "train_AD=", "{:.5f}".format(train_AD),
      "val_MSE=", "{:.5f}".format(val_MSE),
      "val_AD=", "{:.5f}".format(val_AD),
      "time=", "{:.5f}".format(time.time() - t))
    
    if (epoch+1) % save_step == 0:
        test_MSE, test_AD = sess.run((test_mse, test_absolute_diff), 
                                            feed_dict=feed_dict_val) 
        print("-------test_MSE=", "{:.5f}".format(test_MSE),
          "test_AD=", "{:.5f}".format(test_AD))
        