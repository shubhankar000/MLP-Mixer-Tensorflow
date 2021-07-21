import tensorflow as tf


class Projection(tf.keras.layers.Layer):
    """
    Projection Layer to reduce dimensionality of inputs.
    Current implementation only reduces 3d input to 2d.
    Meant to be an alternative to GlobalXXXPooling as a learnable way of "token mixing"
    """

    def __init__(self):
        """
        Resulting projection will be take the size of axis=-1
        """
        super(Projection, self).__init__()

    def build(self, shape):
        if len(shape) != 3:
            raise ValueError(
                f"Input dimension needs to be (None,patches,channels), got {shape} instead"
            )
        self.w = self.add_weight(
            name="w",
            shape=(shape[1], 1),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True,
            # regularizer="l2",
        )
        self.b = self.add_weight(
            name="b",
            shape=(shape[2], 1),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, inputs):
        x = tf.matmul(tf.transpose(inputs, perm=[0, 2, 1]), self.w) + self.b
        return tf.reshape(x, (-1, inputs.shape[2]))


class CreatePatches(tf.keras.layers.Layer):
    """
    Implements the `tf.image.extract_patches` function as a TensorFlow Layer.

    Output is a 4D tensor (batch_size, shape, shape, channels) where shape is the no. of patches along any dimension
    This is calculated using `(n-f)/s + 1`, where n is input dim, f is kernel size, s is stride
    The patches along any 1 dimension of the image is `img_width // kernel`. Since this Layer
    only implements square window sizes, the patch dimensions are `(img_width // kernel, img_width // kernel)`

    For example, if the image is (32,32,3) (CIFAR-10 dimensions), then using a kernel of 8 means
    that each box is (8,8) with stride (8,8). This means there are a total of 16 patches, making a
    (4,4) grid of patches, each (8,8) pixels in dimension. The Patches are taken depthwise for all
    channels, and stacks all the results along the channel dimension.

    Read the documentation of `tf.image.extract_patches` to understand more about the input and
    output dimensions
    """

    def __init__(self, kernel, strides=None, rates=1):
        super(CreatePatches, self).__init__()
        self.kernel = kernel
        self.strides = strides if strides else kernel
        self.rates = rates

    def call(self, images):

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.kernel, self.kernel, 1],
            strides=[1, self.strides, self.strides, 1],
            rates=[1, self.rates, self.rates, 1],
            padding="SAME",
        )

        return patches


class PerPatchFullyConnected(tf.keras.layers.Layer):
    def __init__(self, projection_dims, initial_channels=3):
        """
        projection_dims -- hidden dimension of projection layer
        initial_channels -- Initial channels of the input image. This is needed to apply the same weights to all channels simultaneously
        """
        super(PerPatchFullyConnected, self).__init__()
        self.dims = projection_dims
        self.init_channels = initial_channels

    def build(self, shape):
        assert shape[-1] % self.init_channels == 0
        self.num_patches = shape[1] * shape[2]
        self.w = self.add_weight(
            name="PPFCw",
            shape=(shape[-1], self.dims),
            dtype="float32",
            initializer=tf.keras.initializers.glorot_uniform(),
            trainable=True,
        )
        self.b = self.add_weight(
            name="PPFCb",
            shape=(self.dims,),
            dtype="float32",
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        x = tf.reshape(inputs, (-1, self.num_patches, inputs.shape[-1]))
        return tf.matmul(x, self.w) + self.b


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, initializer="glorot_normal", regularizer=None):
        super(MLPBlock, self).__init__()
        self.ln1 = tf.keras.layers.LayerNormalization(name="ln1")
        self.ln2 = tf.keras.layers.LayerNormalization(name="ln2")
        self.init = initializer
        self.reg = regularizer

    def build(self, shape):
        self.w1 = self.add_weight(
            name="b1w1",
            shape=(shape[-2], shape[-2]),
            dtype="float32",
            initializer=self.init,
            regularizer=self.reg,
            trainable=True,
        )
        self.b1 = self.add_weight(
            name="b1b1", shape=(shape[-2],), initializer="zeros", trainable=True
        )
        self.w2 = self.add_weight(
            name="b1w2",
            shape=(shape[-2], shape[-2]),
            dtype="float32",
            initializer=self.init,
            regularizer=self.reg,
            trainable=True,
        )
        self.b2 = self.add_weight(
            name="b1b2", shape=(shape[-2],), initializer="zeros", trainable=True
        )
        self.w3 = self.add_weight(
            name="b2w1",
            shape=(shape[-1], shape[-1]),
            dtype="float32",
            initializer=self.init,
            regularizer=self.reg,
            trainable=True,
        )
        self.b3 = self.add_weight(
            name="b2b1", shape=(shape[-1],), initializer="zeros", trainable=True
        )
        self.w4 = self.add_weight(
            name="b2w2",
            shape=(shape[-1], shape[-1]),
            dtype="float32",
            initializer=self.init,
            regularizer=self.reg,
            trainable=True,
        )
        self.b4 = self.add_weight(
            name="b2b2", shape=(shape[-1],), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        x = self.ln1(inputs)
        x = tf.transpose(x, perm=[0, 2, 1])

        x = tf.matmul(x, self.w1) + self.b1
        x = tf.nn.gelu(x)
        x = tf.matmul(x, self.w2) + self.b2

        skip = tf.transpose(x, perm=[0, 2, 1]) + inputs

        x = self.ln2(skip)
        x = tf.matmul(x, self.w3) + self.b3
        x = tf.nn.gelu(x)
        x = tf.matmul(x, self.w4) + self.b4 + skip

        return x
