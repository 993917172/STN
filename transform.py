import numpy as np
import tensorflow as tf
import os

def _repeat(x, num_repeat):
    rep  = tf.ones((1, num_repeat), tf.int32)
    x = tf.reshape(x, [-1, 1])
    x = tf.reshape(tf.matmul(x, rep), [-1])
    return x

def transform(theta, img, out_size):
    # theta : [batch_size, 6]
    #  img : [batch_size, hs, ws, c]
    # out_size : [batch_size, ht, wt, c]

    [batch_size, ht, wt, channel] = out_size
    #generate grid
    x = np.linspace(-1, 1, wt)
    y = np.linspace(-1, 1, ht)
    x_s, y_s = np.meshgrid(x, y)
    z_s = np.ones(np.prod(x_s.shape))
    sample_grid = np.vstack((x_s.flatten(), y_s.flatten(), z_s))
    sample_grid = np.resize(sample_grid, (batch_size, 3, ht * wt))
    sample_grid = sample_grid.astype('float32')
    #sample_grid = tf.convert_to_tensor(sample_grid)

    #apply affline to the sample_grid
    theta = tf.reshape(theta, (batch_size, 2, 3))
    batch_grid = tf.matmul(theta, sample_grid)
    batch_grid = tf.transpose(batch_grid, (0, 2, 1)) # [batch_size, ht*wt, 2]

    #bilinear sample
    hs = img.get_shape()[1]
    ws = img.get_shape()[2]
    hs_f = tf.cast(hs, 'float32')
    ws_f = tf.cast(ws, 'float32')
    x = batch_grid[..., 0]
    y = batch_grid[..., 1]
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])
    x = (x + 1) * ws_f * 0.5
    y = (y + 1) * hs_f * 0.5
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, ws-1) #[batch_size * ht * wt]
    x1 = tf.clip_by_value(x1, 0, ws-1)
    y0 = tf.clip_by_value(y0, 0, hs-1)
    y1 = tf.clip_by_value(y1, 0, hs-1)

    dim2 = ws
    dim1 = ws * hs

    base = _repeat(tf.range(batch_size) * dim1, ht*wt)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y0 + x1
    idx_c = base_y1 + x0
    idx_d = base_y1 + x1

    im_flat = tf.reshape(img, [-1, channel])
    Ia = tf.gather(im_flat, idx_a) # [batch_size*ht*wt, 3]
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    wa = tf.expand_dims((x1_f - x) * (y1_f - y), axis=1)
    wb = tf.expand_dims((x - x0_f) * (y1_f - y), axis=1)
    wc = tf.expand_dims((x1_f - x) * (y - y0_f), axis=1)
    wd = tf.expand_dims((x - x0_f) * (y - y0_f), axis=1)

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    out = tf.reshape(out, (batch_size, ht, wt, 3))
    return out











