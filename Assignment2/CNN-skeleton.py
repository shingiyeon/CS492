import tensorflow as tf
import numpy as np
import time
import math

tf.logging.set_verbosity(tf.logging.INFO)

batch_size = 100

def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        # 6 was used in the paper.
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def make_onehot_vector(labels):
    length = len(labels)
    a = np.zeros((length, 10))
    
    for i in range(length):
        a[i][labels[i]] = 1.0
        
    return a
	
def cnn(input, filters, kernel_size, pool_size, strides):
	conv1 = tf.layers.conv2d(
		inputs=input,
		filters=filters,
		kernel_size=kernel_size,
		padding="same",
		activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool_size, strides=strides)
	return pool1

def dense_batch_relu(x, phase, unit, scope, dropout_rate=0.30):
    with tf.variable_scope(scope):
        reg = tf.contrib.layers.l2_regularizer(scale=0.005)
        l1 = tf.layers.dense(x, unit, activation=None, kernel_regularizer=reg)
        l2 = tf.contrib.layers.batch_norm(l1, center=True, scale=True)
#         , is_training=phase
        l3 = tf.layers.dropout(l2, dropout_rate, training=phase)
        
        return tf.nn.relu(l3, 'relu')

def custom_model_fn(features, labels, mode):
    """Model function for PA1"""

    # Write your custom layer

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) # You also can use 1 x 784 vector
#    input_layer = features['x']
    check_train = (mode == tf.estimator.ModeKeys.TRAIN)
    
    '''Hidden Layer'''
    
    L1 = cnn(input_layer, 32, [5,5], [2,2], 2)
    L2 = cnn(L1, 64, [5,5], [2,2], 2)
    L2_flat = tf.reshape(L2, [-1, 7*7*64])
    L3 = dense_batch_relu(L2_flat, check_train, 1024, 'L3')
	
	
    #L1_unit = 500
    #L2_unit = 256
   # L3_unit = 196
   # L4_unit = 128
    
  #  L1 = dense_batch_relu(input_layer, check_train, L1_unit, 'L1')
 #   L2 = dense_batch_relu(L1, check_train, L2_unit, 'L2')
 #   L3 = dense_batch_relu(L2, check_train, L3_unit, 'L3')
#    L4 = dense_batch_relu(L3, check_train, L4_unit, 'L4')
    
    # Output logits Layer
    logits = tf.layers.dense(inputs=L3, units=10, activation=None)
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In predictions, return the prediction value, do not modify
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Select your loss and optimizer from tensorflow API
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) + tf.losses.get_regularization_loss()
#     loss = tf.losses.softmax_cross_entropy(labels["y_one"], logits) # Refer to tf.losses

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(0.01) # Refer to tf.train
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    Y = tf.argmax(labels, 1)
    print(Y)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=Y, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    # Write your dataset path
    dataset_train = np.load('./train.npy')
    dataset_eval =  np.load('./valid.npy')
    test_data =  np.load('./test.npy')

    train_data = dataset_train[:,:784]
    train_labels = dataset_train[:,784].astype(np.int32)
    train_labels_onehot = make_onehot_vector(train_labels)
    
    eval_data = dataset_eval[:,:784]
    eval_labels = dataset_eval[:,784].astype(np.int32)
    eval_labels_onehot = make_onehot_vector(eval_labels)
    
    # Save model and checkpoint
    classifier = tf.estimator.Estimator(model_fn=custom_model_fn, model_dir="./cnn_model81")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=2000)

    # Train the model. You can train your model with specific batch size and epoches
    train_input = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
        y=train_labels_onehot, batch_size=batch_size, num_epochs=5, shuffle=True)
    train_spec = tf.estimator.TrainSpec(input_fn = train_input, max_steps=2501)
#     , hooks=[logging_hook]
#     classifier.train(input_fn=train_input, steps=501, hooks=[logging_hook])

    # Eval the model. You can evaluate your trained model with validation data
    eval_input = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
        y=eval_labels_onehot, num_epochs=1, shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn = eval_input, throttle_secs = 10)
    
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    print("evaluation")
    eval_results = classifier.evaluate(input_fn=eval_input)
    print(eval_results)


    ## ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    pred_input = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, shuffle=False)
    pred_results = classifier.predict(input_fn=pred_input)
    pred_list = list(pred_results)
    result = np.asarray([list(x.values())[1] for x in pred_list])
    ## ----------------------------------------- ##

    np.save('20183309-cnn0.npy', result)
7