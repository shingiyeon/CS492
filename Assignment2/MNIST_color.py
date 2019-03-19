import tensorflow as tf
import numpy as np
import math

tf.logging.set_verbosity(tf.logging.INFO)

batch_size = 50

def make_onehot_vector(labels): 
    length = len(labels)
    a = np.zeros((length, 10))
    
    for i in range(length):
        a[i][labels[i]] = 1.0
        
    return a

def dense_batch_relu(x, phase, unit, scope, dropout_rate=0.30):
    with tf.variable_scope(scope):
        reg = tf.contrib.layers.l2_regularizer(scale=0.005)
        l1 = tf.layers.dense(x, unit, activation=None, kernel_regularizer=reg)
        l2 = tf.contrib.layers.batch_norm(l1, center=True, scale=True)
        l3 = tf.layers.dropout(l2, dropout_rate, training=phase)
        
        return tf.nn.relu(l3, 'relu')

def cnn(input_layer, filters, kernel_size, pool_size, strides, phase, dropout_rate=0.2):
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool_size, strides=strides)
    dropout_layer = tf.layers.dropout(pool1, dropout_rate, training=phase)
    return dropout_layer
    
def augmentation(images, conversion=False, rotater=True):
    transforms_list = []
    c = tf.constant(28.0, dtype=tf.float32)
    original = tf.constant([1,0,0,0,1,0,0,0], dtype=tf.float32)
    if rotater == True:
        angles = tf.random_normal([batch_size], -math.pi, math.pi) 
        transforms_list.append( tf.contrib.image.angles_to_projective_transforms(angles, c, c))
    if conversion==True:
        binomial_prob = tf.less(tf.random_uniform([batch_size], -1.0, 1.0), 0.0) 
        flip_transform = tf.convert_to_tensor( [1, 0, 0, 0, -1, c, 0, 0], dtype=tf.float32) 
        transforms_list.append( tf.where(binomial_prob, tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]), 
                     tf.tile(tf.expand_dims(original, 0), [batch_size, 1]))) 
    images = tf.contrib.image.transform(images, tf.contrib.image.compose_transforms(*transforms_list), interpolation='NEAREST')
    return images


def custom_model_fn(features, labels, mode):
    """Model function for PA2 Extra2"""
    with tf.device('/cpu:0'):
    # Input Layer
        input_layer = tf.reshape(features['x'], [-1, 32, 32, 3])
        print(input_layer)
        check_train = (mode == tf.estimator.ModeKeys.TRAIN)
    #     float(i)
        if check_train == True:
            augmented_data = list()
            augmented_data.append(augmentation(input_layer, conversion=True, rotater=False))
            for i in range(1, 8):
                augmented_data.append(augmentation(input_layer, conversion=True))
            for i in range(1, 8):
                augmented_data.append(augmentation(input_layer, conversion=False))            
            for i in range(0, 15):
                input_layer = tf.concat([input_layer, augmented_data[i]], 0)
            labels = tf.concat([labels, labels], 0)
            labels = tf.concat([labels, labels], 0)
            labels = tf.concat([labels, labels], 0)
            labels = tf.concat([labels, labels], 0)

        input_layer = tf.reshape(input_layer, [-1, 32, 32, 3])
        print(input_layer)

    '''Hidden Layer'''
    with tf.device('/gpu:0'):

        L1 = cnn(input_layer, 32, [5,5], [2,2], 1, phase=check_train)
        print(L1)
        L2 = cnn(L1, 64, [5,5], [2,2], 1, phase=check_train)
        print(L2)
        L3 = cnn(L2, 96, [5,5], [2,2], 2, phase=check_train)
        print(L3)
        L4 = cnn(L3, 128, [5,5], [2,2], 2, phase=check_train)
        print(L4)
        L5 = cnn(L4, 196, [5,5], [2,2], 1, phase=check_train)
        print(L5)
        L5_flat = tf.reshape(L5, [-1, 196*6*6])
        L6 = dense_batch_relu(L5_flat, check_train, 1024, 'L2')
        # Output logits Layer
        logits = tf.layers.dense(inputs=L6, units=10, activation=None)

        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) + tf.losses.get_regularization_loss()

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(0.0005) # Refer to tf.train
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    Y = tf.argmax(labels, 1)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=Y, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



if __name__ == '__main__':
    # Write your dataset path
    dataset_train = np.load('./extra2-train.npy')
    dataset_eval =  np.load('./extra2-valid.npy')
    test_data =  np.load('./extra2-test_img.npy')

    f_dim = dataset_train.shape[1] - 1
    train_data = dataset_train[:,:f_dim].astype(np.float32)
    print(train_data[0][0])
    train_labels = dataset_train[:,f_dim].astype(np.int32)
    train_labels_onehot = make_onehot_vector(train_labels)
    
    eval_data = dataset_eval[:,:f_dim].astype(np.float32)
    eval_labels = dataset_eval[:,f_dim].astype(np.int32)
    eval_labels_onehot=make_onehot_vector(eval_labels)
    
    test_data = test_data.astype(np.float32)

    # Save model and checkpoint
    mnist_classifier = tf.estimator.Estimator(model_fn=custom_model_fn, model_dir="./extra2_layer7")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1000)

    # Train the model. You can train your model with specific batch size and epoches
    train_input = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
        y=train_labels_onehot, batch_size=50, num_epochs=5, shuffle=True)
    train_spec = tf.estimator.TrainSpec(input_fn = train_input, max_steps=8001)

    # Eval the model. You can evaluate your trained model with validation data
    eval_input = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
        y=eval_labels_onehot, num_epochs=1, shuffle=False)
    
    eval_spec = tf.estimator.EvalSpec(input_fn = eval_input, throttle_secs = 30)
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
    print("evaluation")
    eval_results = mnist_classifier.evaluate(input_fn=eval_input)
    print(eval_results)


    ## ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    pred_input = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, shuffle=False)
    pred_results = mnist_classifier.predict(input_fn=pred_input)
    pred_list = list(pred_results)
    result = np.asarray([list(x.values())[1] for x in pred_list])
    ## ----------------------------------------- ##

    np.save('extra2_20183309_layer7.npy', result)
