=====
Usage
=====

qlty provides tools unstitch and stitch pyutorch tensors.

To use qlty in a project import it::

    import qlty
    from qlty import qlty2D

Lets make some mock data::

    import einops
    import torch
    import numpy as np

    x = torch.rand((10,3,128,128))
    y = torch.rand((10,1,128,128))

Assume that x and y are data wwhose relation you are trying to learn using some network, such that after training,
you have something like::

    y_guess = net(x)

with::

    torch.sum( torch.abs(y_guess - y) ) < a_small_number

If the data you have is large and doesn't fit onto your GPU card, or if you need to chop things up into smaller bits
for boundary detection qlty can be use. Lets take the above data and chop it into smaller bits::

    quilt = qlty2D.NCYXQuilt(X=128,
                             Y=128,
                             window=(16,16),
                             step=(4,4),
                             border=(4,4),
                             border_weight=0)

This object now allows one to cut any input tensor with shape (N,C,Y,X) into smaller, overlapping patches of size
(M,C,Ywindow,Xwindow). The moving window, in this case a 16x16 patch, is moved along the input tensor with steps
(4,4). In addition, we define a border region in these patches of 4 pixels wide. Pixels in this area will we assigned
weight border_weight (0 in this case) when data is stitched back together (more later).

Lets unstitch the (x,y) training data pair::

    x_bits, y_bits = quilt.unstitch_data_pair(x,y)
    print("x shape: ",x.shape)
    print("y shape: ",y.shape)
    print("x_bits shape:", x_bits.shape)
    print("y_bits shape:", y_bits.shape)

Yielding::

    x shape:  torch.Size([10, 3, 128, 128])
    y shape:  torch.Size([10, 128, 128])
    x_bits shape: torch.Size([8410, 3, 16, 16])
    y_bits shape: torch.Size([8410, 16, 16])


If we now make some mock data that a neural network has returned:

    y_mock = torch.rand( (8410,17,16,16))

we can sticth it back together into the right shape, averaging overlapping areas, excluding or downweighting border
areas::

    y_stiched, weights = quilt.stitch(y_mock)

which gives::

    print(y_stiched.shape)
    torch.Size([10, 17, 128, 128])

The 'weights' tensor encodes how many contributors there were for each pixel.
