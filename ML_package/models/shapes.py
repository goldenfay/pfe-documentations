CSRNET_FRONTEND = [

    ('C2D', {
        'in_channels': 3,
        'out_channels': 64,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 64,
        'out_channels': 64,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('M', {
        'ks': 2,
        'stride': 2,

    }),
    ('C2D', {
        'in_channels': 64,
        'out_channels': 128,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 128,
        'out_channels': 128,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('M', {
        'ks': 2,
        'stride': 2,

    }),
    ('C2D', {
        'in_channels': 128,
        'out_channels': 256,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 256,
        'out_channels': 256,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 256,
        'out_channels': 256,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('M', {
        'ks': 2,
        'stride': 2,

    }),
    ('C2D', {
        'in_channels': 256,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 512,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 512,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    })


]

CSRNET_BACKEND = [
    ('C2D', {
        'in_channels': 512,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 512,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 512,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 512,
        'out_channels': 256,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 256,
        'out_channels': 128,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 128,
        'out_channels': 64,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    })
]
