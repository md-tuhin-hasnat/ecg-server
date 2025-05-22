import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="MyModels")
class detector(Model):
    def __init__(self, **kwargs):
        super(detector, self).__init__(**kwargs)  # Pass built-in arguments like trainable, dtype

        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(140, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        config = super().get_config()
        # Add custom config if needed
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
