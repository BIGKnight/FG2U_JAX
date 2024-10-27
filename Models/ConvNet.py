import jax
import haiku as hk

class ConvNet(hk.Module):
    def __init__(self, num_classes, net_depth=3, net_width=128, channel=3,
                 net_act='relu', net_pooling='avg', net_norm='instance', im_size=(32, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.net_depth = net_depth
        self.net_width = net_width
        self.channel = channel
        self.net_act = net_act
        self.net_pooling = net_pooling
        self.net_norm = net_norm
        self.im_size = im_size


    def __call__(self, x, is_training=True):
        # Dynamically construct layers based on depth
        for d in range(self.net_depth):
            # print('Depth:', d, 'conv_x shape:', x.shape)
            x = hk.Conv2D(self.net_width, kernel_shape=3, stride=1, padding='SAME')(x)
            # print('Depth:', d, 'conv_x shape:', x.shape)
            x = self._apply_activation(x)
            # print('Depth:', d, 'act_x shape:', x.shape)
            x = self._apply_norm(x, is_training)
            # print('Depth:', d, 'IN_x shape:', x.shape)
            x = self._apply_pooling(x)
            # print('Depth:', d, 'POOL_x shape:', x.shape)
        
        x = hk.Flatten()(x)
        x = hk.Linear(self.num_classes)(x)
        return x
    
    def _apply_activation(self, x):
        if self.net_act == 'relu':
            return jax.nn.relu(x)
        elif self.net_act == 'leakyrelu':
            return jax.nn.leaky_relu(x, negative_slope=0.01)
        elif self.net_act == 'sigmoid':
            return jax.nn.sigmoid(x)
        else:
            raise ValueError("Unknown activation function")

    def _apply_pooling(self, x):
        if self.net_pooling == 'avg':
            return hk.avg_pool(x, window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
        elif self.net_pooling == 'max':
            return hk.max_pool(x, window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
        elif self.net_pooling == 'none':
            return x
        else:
            raise ValueError("Unknown pooling method")

    def _apply_norm(self, x, is_training):
        if self.net_norm == 'batch':
            return hk.BatchNorm(True, True, 0.99)(x, is_training)
        elif self.net_norm == 'instance':
            return hk.InstanceNorm(True, True)(x)
        elif self.net_norm == 'none':
            return x
        else:
            raise ValueError("Unknown normalization method")

def model_fn(batch, num_classes, net_depth, net_width, channel, net_act, net_pooling, net_norm, im_size):
    model = ConvNet(num_classes, net_depth, net_width, channel, net_act, net_pooling, net_norm, im_size)
    return model(batch)

# Example usage:
# Initialize model
# model = hk.transform(lambda x: model_fn(x, num_classes=10, net_depth=3, ...))
# params = model.init(jax.random.PRNGKey(42), jnp.ones([1, 32, 32, 3]))
# logits = model.apply(params, None, jnp.ones([1, 32, 32, 3]))
