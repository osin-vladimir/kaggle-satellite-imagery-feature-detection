import mxnet as mx

def print_inferred_shape(net, name):
    print name
    ar, ou, au = net.infer_shape(data=(1, 1, 400, 400))
    print ou

def ru(data, num_filter, bn_mom=0.9, count=1):

    for i in range(count):
        net = mx.sym.BatchNorm(data, fix_gamma=False, eps=2e-5, momentum=bn_mom)
        net = mx.sym.LeakyReLU(net, act_type='prelu')
        net = mx.sym.Convolution(net, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)

        net = mx.sym.BatchNorm(net, fix_gamma=False, eps=2e-5, momentum=bn_mom)
        net = mx.sym.LeakyReLU(net, act_type='prelu')
        net = mx.sym.Convolution(net, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)

        net = mx.sym.BatchNorm(net, fix_gamma=False, eps=2e-5, momentum=bn_mom)
        net = mx.sym.LeakyReLU(net, act_type='prelu')
        net = mx.sym.Convolution(net, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)
        net = net + data
        data = net

    return data


def residual_unit_2(data, num_filter, bn_mom=0.9):
    net = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5)
    net = mx.sym.LeakyReLU(data=net, act_type='prelu')
    net = mx.sym.Convolution(data=net, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)

    net = mx.sym.BatchNorm(data=net, fix_gamma=False, momentum=bn_mom, eps=2e-5)
    net = mx.sym.LeakyReLU(data=net, act_type='prelu')
    net = mx.sym.Convolution(data=net, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)

    return net + data

def bna(net):
    # net = mx.symbol.BatchNorm(net)
    net = mx.symbol.LeakyReLU(net, act_type="elu")
    return net

def conv_bna(net, num_filter, is_pool=False):

    if is_pool:
        net = mx.symbol.Convolution(net, num_filter=num_filter, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    else:
        net = mx.symbol.Convolution(net, num_filter=num_filter, kernel=(3, 3), pad=(1, 1))

    net = bna(net)
    return net

def up_bna(net, net_merge, num_filter, num_filter_up, up_type='deconv'):
    net = conv_bna(net, num_filter)
    net = conv_bna(net, num_filter)

    if up_type == 'upsample':
        # Nearest Neighbor is best used for categorical data like land-use classification or slope classification.
        # The values that go into the grid stay exactly the same, a 2 comes out as a 2 and 99 comes out as 99.
        # The value of of the output cell is determined by the nearest cell center on the input grid.
        # Nearest Neighbor can be used on continuous data but the results can be blocky.
        net = mx.sym.UpSampling(net, scale=2, num_filter=num_filter_up, sample_type='nearest')
    elif up_type ==  'deconv':
        net = mx.sym.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=num_filter_up)

    net = mx.symbol.Concat(net, net_merge, num_args=2, dim=1)
    net = bna(net)
    return net


def get_unet_symbol():
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    # group 1
    net = conv_bna(data, 64)
    net_merge_1 = conv_bna(net, 64)
    net = conv_bna(net_merge_1, 64, is_pool=True)

    # group 2
    net = conv_bna(net, 128)
    net_merge_2= conv_bna(net, 128)
    net = conv_bna(net_merge_2, 128, is_pool=True)

    # group 3
    net = conv_bna(net, 256)
    net_merge_3 = conv_bna(net, 256)
    net = conv_bna(net_merge_3, 256, is_pool=True)

    # group 4
    net = conv_bna(net, 512)
    net_merge_4 = conv_bna(net, 512)
    net = conv_bna(net_merge_4, 512, is_pool=True)

    # up groups
    net = up_bna(net, net_merge_4, 1024, 512)
    net = up_bna(net, net_merge_3, 512, 256)
    net = up_bna(net, net_merge_2, 256, 128)
    net = up_bna(net, net_merge_1,   128, 64)

    # final group
    net = conv_bna(conv_bna(net, 64), 64)
    net = mx.symbol.Convolution(net, num_filter=1, kernel=(1, 1))
    sigmoid = mx.symbol.Activation(net, act_type='sigmoid', name='sigmoid')

    return mx.sym.softmax_cross_entropy(sigmoid, label)


