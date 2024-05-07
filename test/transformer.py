####
###  Model from keras
###  Takes 3D input (np.array) and returns 3D output (np.array)
import keras
from keras import layers

class ITPF_Transformer():
    def transformer_encoder(
            self,
            inputs, 
            head_size, 
            num_heads, 
            ff_dim, 
            dropout=0
        ):
        # Attention and Normalization
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="sigmoid")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
    
    def build_model(
            self,
            input_shape,
            output_shape,
            output_features_vector,
            head_size,
            num_heads,
            ff_dim,
            num_transformer_blocks,
            mlp_units,
            dropout=0,
            mlp_dropout=0,
        ):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        
        # output layer
        outputs = layers.Dense(output_features_vector)(x)
        outputs = layers.Reshape(output_shape)(outputs)
        return keras.Model(inputs, outputs)
