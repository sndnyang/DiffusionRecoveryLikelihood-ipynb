"""
Network structure of the energy-based mdoels
"""

from config import FLAGS
import tensorflow.compat.v2 as tf
import tensorflow_addons as tfa
import nn


def nonlinearity(x, act):
  if act == 'lrelu':
    return tf.nn.leaky_relu(x)
  if act == 'swish':
    return tf.nn.swish(x)

  raise NotImplementedError


class normalize(tf.keras.layers.Layer):
  def __init__(self, name, *args, **kwargs):
    super(normalize, self).__init__(name=name)
    self.normalize = kwargs['normalize']
    self.norm = None
    if self.normalize:
      if self.normalize == 'group_norm':
        self.norm = tfa.layers.GroupNormalization(groups=32, epsilon=1e-06)
      elif self.normalize == 'batch_norm':
        self.norm = tf.keras.layers.BatchNormalization()
      elif self.normalize == 'instance_norm':
        self.norm = tfa.layers.InstanceNormalization()

  def call(self, inputs, **kwargs):
    if self.norm is not None:
      inputs = self.norm(inputs, training=True)
    return inputs


class downsample(tf.keras.layers.Layer):
  def __init__(self, name, hps, with_conv, *args, **kwargs):
    super(downsample, self).__init__(name=name)
    self.with_conv = with_conv
    self.hps = hps

  def build(self, input_shape):
    B, H, W, C = input_shape
    if self.with_conv:
      self.conv2d = nn.conv2d(name='conv', num_units=C, filter_size=3, stride=2, spec_norm=self.hps.spec_norm)
    # print('{}: x={}'.format(self.name, input_shape))

  def call(self, inputs, **kwargs):
    B, H, W, C = inputs.shape

    if self.with_conv:
      x = self.conv2d(inputs)
    else:
      x = tf.nn.avg_pool(inputs, 2, 2, 'SAME')
    assert x.shape == [B, H // 2, W // 2, C]

    return x


class resnet_block(tf.keras.layers.Layer):
  def __init__(self, *, name, hps, out_ch=None):
    super(resnet_block, self).__init__(name=name)
    self.out_ch = out_ch
    self.conv_shortcut = hps.res_conv_shortcut
    self.spec_norm = hps.spec_norm
    self.use_scale = hps.res_use_scale
    self.hps = hps

  def build(self, input_shape):
    B, H, W, C = input_shape
    if self.out_ch is None:
      self.out_ch = C
    self.normalize_1 = normalize('norm1', normalize=self.hps.normalize)
    self.normalize_2 = normalize('norm2', normalize=self.hps.normalize)

    self.dense = nn.dense(name='temb_proj', num_units=self.out_ch, spec_norm=self.spec_norm)
    self.conv2d_1 = nn.conv2d(name='conv1', num_units=self.out_ch, spec_norm=self.spec_norm)

    self.conv2d_2 = nn.conv2d(
      name='conv2', num_units=self.out_ch, init_scale=0., spec_norm=self.spec_norm, use_scale=self.use_scale
    )
    if self.conv_shortcut:
      self.conv2d_shortcut = nn.conv2d(name='conv_shortcut', num_units=self.out_ch, spec_norm=self.spec_norm)
    else:
      self.nin_shortcut = nn.nin(name='nin_shortcut', num_units=self.out_ch, spec_norm=self.spec_norm)
    # print('{}: x={}'.format(self.name, input_shape))

  def call(self, inputs, temb=None, dropout=0.):
    B, H, W, C = inputs.shape
    x = inputs
    h = inputs

    h = nonlinearity(self.normalize_1(h), self.hps.act)
    h = self.conv2d_1(h)

    if temb is not None:
      # add in timestep embedding
      h += self.dense(nonlinearity(temb, self.hps.act))[:, None, None, :]

    h = nonlinearity(self.normalize_2(h), self.hps.act)
    h = tf.nn.dropout(h, rate=dropout)
    h = self.conv2d_2(h)

    if C != self.out_ch:
      if self.conv_shortcut:
        x = self.conv2d_shortcut(x)
      else:
        x = self.nin_shortcut(x)

    assert x.shape == h.shape
    return x + h


class attn_block(tf.keras.layers.Layer):
  def __init__(self, name):
    super(attn_block, self).__init__(name=name)

  def build(self, input_shape):
    B, H, W, C = input_shape
    self.normalize = normalize(name='norm', normalize=self.hps.normalize)
    self.nin_q = nn.nin(name='q', num_units=C)
    self.nin_k = nn.nin(name='k', num_units=C)
    self.nin_v = nn.nin(name='v', num_units=C)

    self.nin_proj_out = nn.nin(name='proj_out', num_units=C, init_scale=0.)
    # print('{}: x={}'.format(self.name, input_shape))

  def call(self, inputs):
    x = inputs
    B, H, W, C = x.shape

    h = self.normalize(x)
    q = self.nin_q(h)
    k = self.nin_k(h)
    v = self.nin_v(h)

    w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = tf.reshape(w, [B, H, W, H * W])
    w = tf.nn.softmax(w, -1)
    w = tf.reshape(w, [B, H, W, H, W])

    h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
    h = self.nin_proj_out(h)

    assert h.shape == x.shape

    return x + h


class net_res_temb2(tf.keras.layers.Layer):
  def __init__(self, *, name, ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
          attn_resolutions, hps):
    super(net_res_temb2, self).__init__(name=name)
    self.ch, self.ch_mult = ch, ch_mult
    self.num_res_blocks = num_res_blocks
    self.attn_resolutions = attn_resolutions
    self.num_resolutions = len(self.ch_mult)
    self.resamp_with_conv = hps.resamp_with_conv
    self.use_attention = hps.use_attention
    self.spec_norm = hps.spec_norm
    self.hps = hps

  def build(self, input_shape):
    # timestep embedding
    self.temb_dense_0 = nn.dense(name='temb/dense0', num_units=self.ch * 4, spec_norm=self.spec_norm)
    self.temb_dense_1 = nn.dense(name='temb/dense1', num_units=self.ch * 4, spec_norm=self.spec_norm)
    self.temb_dense_2 = nn.dense(name='temb/dense2', num_units=self.ch * self.ch_mult[-1], spec_norm=False)

    S = input_shape[-3]
    self.res_levels = []
    self.attn_s = dict()
    self.downsample_s = []

    # downsample
    self.conv2d_in = nn.conv2d(name='conv_in', num_units=self.ch, spec_norm=self.spec_norm)
    for i_level in range(self.num_resolutions):
      res_s = []
      if self.use_attention and S in self.attn_resolutions:
        self.attn_s[str(S)] = []
      for i_block in range(self.num_res_blocks):
        res_s.append(
          resnet_block(
            name='level_{}_block_{}'.format(i_level, i_block), hps=self.hps, out_ch=self.ch * self.ch_mult[i_level]
          )
        )
        if self.use_attention and S in self.attn_resolutions:
            self.attn_s[str(S)].append(attn_block(name='down_{}_attn_{}'.format(i_level, i_block)))
      self.res_levels.append(res_s)

      if i_level != self.num_resolutions - 1:
        self.downsample_s.append(downsample(name='downsample_{}'.format(i_level), hps=self.hps, with_conv=self.resamp_with_conv))
        S = S // 2

    # end
    self.normalize_out = normalize(name='norm_out', normalize=self.hps.normalize)
    self.fc_out = nn.dense(name='dense_out', num_units=1, spec_norm=False)

  def call(self, inputs, t, dropout):
    x = inputs
    B, S, _, _ = x.shape
    assert x.dtype == tf.float32 and x.shape[2] == S
    if isinstance(t, int) or len(t.shape) == 0:
      t = tf.ones([B], dtype=tf.int32) * t

    # Timestep embedding
    temb = nn.get_timestep_embedding(t, self.ch)
    temb = self.temb_dense_0(temb)
    temb = self.temb_dense_1(nonlinearity(temb, self.hps.act))
    assert temb.shape == [B, self.ch * 4]

    # downsample
    h = self.conv2d_in(x)
    for i_level in range(self.num_resolutions):
      for i_block in range(self.num_res_blocks):
        h = self.res_levels[i_level][i_block](h, temb=temb, dropout=dropout)

        if self.use_attention:
          if h.shape[1] in self.attn_resolutions:
            h = self.attn_s[str(h.shape[1])][i_block](h)

      if i_level != self.num_resolutions - 1:
        h = self.downsample_s[i_level](h)

    # end
    if self.hps.final_act == 'relu':
      h = tf.nn.relu(h)
    elif self.hps.final_act == 'swish':
      h = tf.nn.swish(h)
    elif self.hps.final_act == 'lrelu':
      tf.nn.leaky_relu(x)
    else:
      raise NotImplementedError
    h = tf.reduce_sum(h, [1, 2])
    temb_final = self.temb_dense_2(nonlinearity(temb, self.hps.act))
    h = tf.reduce_sum(h * temb_final, axis=1)

    return h