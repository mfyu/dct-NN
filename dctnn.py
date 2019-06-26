import numpy as np
from PIL import Image
import tensorflow as tf
from scipy import fftpack
import math

label = np.loadtxt('combined_label.gz')
data = np.loadtxt('combined_train.gz')

im = Image.open('/mnt/net/filer-ai/shares/proj/pam/data/image/silo/exam/exam.quiz/LCDF000.bmp')
block_shape = [4,4]
original_shape = [1080,1920]
quantize50 = [[16,11,10,16],
              [12,12,14,19],
              [14,13,16,24],
              [14,17,22,29]]
#GPU config
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
def reconstruct(axon, orig_shape, block_shape): #{
    """
    Reconstructs a vector based on its original shape and block shape that it
    was deconstructed with. Can replace bad blocks with alternate data.

    Args:
        axon: A numpy array of data deconstructed with parameter `block_shape`.
        orig_shape: A list or tuple with the image shape to be reconstructed.
        block_shape: A list or tuple with the shape of blocks used when 
            vectorizing the image.
        replace_data: A numpy array to replace bad blocks with.
        bad_indices: An array-like object containing the indices of rows in
            `axon` to be replaced with rows in `replace_data`.

    Returns:
        vec: An array of shape [`shape[0]`, `shape[1]`, 3] of the image vector.
    """
    assert (block_shape[0]*block_shape[1] == axon.shape[1]), f'ReconstructionError: Shape mismatch {block_shape[0]*block_shape[1]} vs {axon.shape[1]}'
    if axon.dtype != 'uint8': #{
        axon = axon.copy()
    #}
    rows, cols   = orig_shape
    brows, bcols = block_shape

    reconstruct = np.zeros(orig_shape, dtype='uint8')
    count = 0
    for i in range(0, rows, brows): #{
        for j in range(0, cols, bcols): #{
            block = axon[count].reshape(block_shape)
            reconstruct[i:i+brows, j:j+bcols] = block
            count += 1
        #}
    #}
    return reconstruct

def mse(original, new):
    orig = np.asarray(original)
    return (np.square(orig - new)).mean()

def psnr(original, new):
    return 10*math.log10(255**2/mse(original,new))
#####################################################################################
#-------------------------FORWARD---------------------------------------------------#
#####################################################################################
def forward(TRAIN_STEPS):
  #  label = np.loadtxt('combined_label.gz')
   # data = np.loadtxt('combined_train.gz')

    test_label = np.loadtxt('frame0_quantized_labels.gz')
    test_data = np.loadtxt('/mnt/net/filer-ai/shares/hotel/scai/dctpredict/2ddctdatagen/frame0_im.gz')

    #reverse
    # data = np.loadtxt('frame39_quantized_labels.gz')
    # label = np.loadtxt('frame39_data.gz')

    # test_data = np.loadtxt('frame0_quantized_labels.gz')
    # test_label = np.loadtxt('frame0_data.gz')
    print(label.shape)
    print(data.shape)

    randomize = np.arange(len(label))
    np.random.shuffle(randomize)
    label = label[randomize]
    data = data[randomize]

    #This method splits the data and label set into training and testing sets, splitting at num
    def TRAIN_TEST_SIZE(num):
        x_train = data[:num,:]
        y_train = label[:num,:]
        print('training set: {}'.format(num))
        x_test = data[num:,:]
        y_test = label[num:,:]
        print('test set: {}'.format(data.shape[0]-num))
        return x_train, y_train, x_test, y_test

    #Setting up the training and testing sets
    x_train, y_train, x_test, y_test = TRAIN_TEST_SIZE(15552000)

    x_test = test_data
    y_test = test_label

    #NN layer sizes
    n_inputs = 16
    layer1 = 16
    n_outputs = 16

    #Starting tensorflow session
    sess = tf.Session(config=config)

    #Initialize tensorflow variables
    x = tf.placeholder(tf.float32, shape=[None, n_inputs])
    y_ = tf.placeholder(tf.float32, shape=[None, n_outputs])

    weights = {
        'out' : tf.Variable(tf.random_normal([n_inputs,n_outputs])),
    }

    biases = {
        'out': tf.Variable(tf.random_normal([n_outputs])),
    }

    initializer = tf.contrib.layers.xavier_initializer()
    W1 = tf.Variable(initializer([16,16]))
    W2 = tf.Variable(initializer([16,16]))
    b1 = tf.Variable(initializer([16]))
    b2 = tf.Variable(initializer([16]))

    #This method creates the structure of the NN
    def NN(x):
        layer1 = tf.add(tf.matmul(x, W1), b1)
       # out = tf.add(tf.matmul(x, weights['out']), biases['out'])
        out = tf.add(tf.matmul(layer1, W2), b2)
        return out

    #Creating the NN
    nn = NN(x)
    y=nn

    #Training constants
    LEARNING_RATE = 0.01
   # TRAIN_STEPS = 100

    correct_prediction = tf.equal(y, y_)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.squared_difference(y,y_))
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    cross_entropy = tf.reduce_mean(tf.squared_difference(y,y_))


    training = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess.run(init)

    #Training and testing begins here
    for i in range(TRAIN_STEPS+1):
         l,_,a = sess.run([cross_entropy, training, accuracy], 
                         feed_dict={x: x_train, y_: y_train})
         if i%100 == 0:
            print('Training Step:' + str(i) + ' MSE =  ' + str(a) + '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})) +  ' Test  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})))


    prediction = sess.run(y, feed_dict={x: x_test})
    dctw = sess.run(W1)
    dctb = sess.run(b1)


    np.savetxt('dct_weights.gz', dctw)
    np.savetxt('dct_biases.gz', dctb)
    np.savetxt('prediction.gz', prediction)
    print(prediction.shape)
    return prediction













#####################################################################################
#-------------------------REVERSE---------------------------------------------------#
#####################################################################################

def reverse(TRAIN_STEPS, test, label, train):# weights, biases):

    #reverse
  #  data = np.loadtxt('combined_label.gz')
   # label = np.loadtxt('combined_train.gz')
    data = train

    test_data = test
    test_label = np.loadtxt('frame0_data.gz')
    print(label.shape)
    print(data.shape)

    randomize = np.arange(len(label))
    np.random.shuffle(randomize)
    label = label[randomize]
    data = data[randomize]

    #This method splits the data and label set into training and testing sets, splitting at num
    def TRAIN_TEST_SIZE(num):
        x_train = data[:num,:]
        y_train = label[:num,:]
        print('training set: {}'.format(num))
        x_test = data[num:,:]
        y_test = label[num:,:]
        print('test set: {}'.format(data.shape[0]-num))
        return x_train, y_train, x_test, y_test

    #Setting up the training and testing sets
    x_train, y_train, x_test, y_test = TRAIN_TEST_SIZE(15552000)

    x_test = test_data
    y_test = test_label

    #NN layer sizes
    n_inputs = 16

    n_layer1 = 100
    n_layer2 = 100
    n_layer3 = 100
    n_outputs = 16

    #Starting tensorflow session
    sess = tf.Session(config=config)

    #Initialize tensorflow variables
    x = tf.placeholder(tf.float32, shape=[None, n_inputs])
    y_ = tf.placeholder(tf.float32, shape=[None, n_outputs])

  #  weights = {
      
   #     'out' : tf.Variable(tf.random_normal([n_inputs,n_outputs])),
   # }

  #  biases = {
     
  #      'out': tf.Variable(tf.random_normal([n_outputs])),
   # }


    initializer = tf.contrib.layers.xavier_initializer()
    W1 = tf.Variable(initializer([16,n_layer1]))
    W2 = tf.Variable(initializer([n_layer1,n_layer2]))
    W3 = tf.Variable(initializer([n_layer2,n_layer3]))
    W4 = tf.Variable(initializer([n_layer3,16]))

    b1 = tf.Variable(initializer([n_layer1]))
    b2 = tf.Variable(initializer([n_layer2]))
    b3 = tf.Variable(initializer([n_layer3]))
    b4 = tf.Variable(initializer([16]))
   # W1 = tf.Variable(dtype=tf.float32,initial_value= weights)
   # b1 = tf.Variable(dtype=tf.float32,initial_value= biases)

    #This method creates the structure of the NN
    def NN(x):
        layer1 = tf.add(tf.matmul(x, W1), b1)
       # out = tf.add(tf.matmul(x, weights['out']), biases['out'])
        layer2 = tf.add(tf.matmul(layer1, W2), b2)
        layer3 = tf.add(tf.matmul(layer2, W3), b3)
        out = tf.add(tf.matmul(layer3, W4), b4)
        return out

    #Creating the NN
    nn = NN(x)
    y=nn

    #Training constants
    LEARNING_RATE = 0.01
   # TRAIN_STEPS = 100

    correct_prediction = tf.equal(y, y_)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.squared_difference(y,y_))
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    cross_entropy = tf.reduce_mean(tf.squared_difference(y,y_))


    training = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess.run(init)

    #Training and testing begins here
    for i in range(TRAIN_STEPS+1):
        l,_,a = sess.run([cross_entropy, training, accuracy], 
                         feed_dict={x: x_train, y_: y_train})
        if i%100 == 0:
            print('Training Step:' + str(i) + ' MSE =  ' + str(a) + '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})) +  ' Test  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})))


    prediction = sess.run(y, feed_dict={x: x_test})

    # prediction = prediction.reshape((-1,4,4))
    # for i in range(len(prediction)):
    #     prediction[i] = prediction[i]*quantize50
    #     prediction[i] = fftpack.idct(prediction[i], norm='ortho')

    r = prediction[0:129600]
    g = prediction[129600:259200]
    b = prediction[259200:388800]


    r = r.round()
    g = g.round()
    b = b.round()
    r = np.clip(r,0,255).astype('uint8')
    g = np.clip(g,0,255).astype('uint8')
    b = np.clip(b,0,255).astype('uint8')

    r = reconstruct(r,original_shape,block_shape)
    g = reconstruct(g,original_shape,block_shape)
    b = reconstruct(b,original_shape,block_shape)

    # r = r.reshape((1080,1920))
    # g = g.reshape((1080,1920))
    # b = b.reshape((1080,1920))

    # r = np.clip(r.reshape((1080,1920)),0,255).astype('uint8')
    # g = np.clip(g.reshape((1080,1920)),0,255).astype('uint8')
    # b = np.clip(b.reshape((1080,1920)),0,255).astype('uint8')
    recon = np.dstack((r,g,b))
    print(f'PSNR: {psnr(im,recon)}')
    psnrfile = open('combineddata/frame0/3hiddenlayer100neuron/psnr.txt', 'a')
    psnrfile.write(f'Train Steps: {TRAIN_STEPS}, PSNR: {psnr(im,recon)}\n')
    psnrfile.close()

    Image.fromarray(np.uint8(recon)).save('combineddata/frame0/3hiddenlayer100neuron/'+str(TRAIN_STEPS)+'step_foward_reverse.bmp')


#prediction = forward(4000)
#w = np.loadtxt('dct_weights.gz')
#b = np.loadtxt('dct_biases.gz')
prediction = np.loadtxt('prediction.gz')

for trainstep in range(100,3100,500):
    reverse(trainstep,prediction,data,label)
