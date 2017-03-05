import mxnet as mx

def get_fcn_symbol():
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    # group 1
    net = mx.symbol.Convolution(data=data, num_filter=32, kernel=(5, 5), pad=(2, 2), dilate=(1, 1), stride=(1, 1), name="conv1")
    net = mx.symbol.BatchNorm(net)
    net = mx.symbol.LeakyReLU(net, act_type="prelu")
    net = mx.symbol.Pooling(net, pool_type='max', stride=(1, 1), kernel=(3, 3), pad=(1, 1), name="pool1")

    # group 2
    net = mx.symbol.Convolution(net, num_filter=96, kernel=(5, 5), pad=(4, 4), dilate=(2, 2), stride=(1, 1), name="conv2")
    net = mx.symbol.BatchNorm(net)
    net = mx.symbol.LeakyReLU(net, act_type="prelu")
    net = mx.symbol.Pooling(net, pool_type='max', stride=(1, 1), kernel=(5, 5), pad=(2, 2), name='pool2')

    # group 3
    net = mx.symbol.Convolution(net, num_filter=128, kernel=(3, 3), pad=(4, 4), dilate=(4, 4), stride=(1, 1), name="conv3")
    net = mx.symbol.BatchNorm(net)
    net = mx.symbol.LeakyReLU(net, act_type="prelu")
    net = mx.symbol.Pooling(net, pool_type='max', stride=(1, 1), kernel=(9, 9), pad=(4, 4), name='pool3')

    # group 4
    net = mx.symbol.Convolution(net, num_filter=128, kernel=(3, 3), pad=(8, 8), dilate=(8, 8), stride=(1, 1), name="conv4")
    net = mx.symbol.BatchNorm(net)
    net = mx.symbol.LeakyReLU(net, act_type="prelu")
    net = mx.symbol.Pooling(net, pool_type='max', stride=(1, 1), kernel=(17, 17), pad=(8, 8), name='pool4')

    # group 5
    net = mx.symbol.Convolution(net, num_filter=512, kernel=(3, 3), pad=(16, 16), dilate=(16, 16), stride=(1, 1), name="fc5")
    net = mx.symbol.BatchNorm(net)
    net = mx.symbol.LeakyReLU(net, act_type="prelu")
    net = mx.symbol.Convolution(net, num_filter=512, kernel=(1, 1), stride=(1, 1), name="fc6")
    net = mx.symbol.BatchNorm(net)
    net = mx.symbol.LeakyReLU(net, act_type="prelu")

    # final layer
    net = mx.symbol.Convolution(net, num_filter=1, kernel=(1, 1), stride=(1, 1), name="fc7")
    net = mx.symbol.BatchNorm(net)
    net = mx.symbol.LeakyReLU(net, act_type="prelu")

    # jaccard index loss
    sigmoid = mx.symbol.Activation(net, act_type="sigmoid", name='fc8')
    intersection = mx.symbol.sum((sigmoid * label), axis=(2, 3))
    summa = mx.symbol.sum((sigmoid+label), axis=(2,3))
    jac = -(intersection/summa)

    return mx.symbol.MakeLoss(jac)






