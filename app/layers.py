from tensorflow.keras.layers import Layer

# Classes that are employed in the custom Siamese model

# Siamese Distance Class
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
