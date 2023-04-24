import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, RepeatVector, LSTM, TimeDistributed, Reshape
import string


chars = string.ascii_letters + string.digits + string.whitespace

input = np.random.rand(1, 80, 200, 3)
print(f"Input => {input.shape}")

l1 = Conv2D(32, (3, 3), input_shape=(80, 200, 3), padding="same")(input)
print(f"Layer 1 conv2d => {l1.shape}")
l2 = MaxPooling2D(pool_size=(2,2))(l1)
print(f"Layer 2 maxPooling => {l2.shape}")

l3 = Conv2D(32, (3, 3), padding="same")(l2)
print(f"Layer 3 conv2d => {l3.shape}")
l4 = MaxPooling2D(pool_size=(2,2))(l3)
print(f"Layer 4 maxPooling => {l4.shape}")

l5 = Conv2D(48, (3, 3), padding="same")(l4)
print(f"Layer 5 conv2d => {l5.shape}")
l6 = MaxPooling2D(pool_size=(2,2))(l5)
print(f"Layer 6 maxPooling => {l6.shape}")

l7 = Conv2D(48, (3, 3), padding="same")(l6)
print(f"Layer 7 conv2d => {l7.shape}")
l8 = MaxPooling2D(pool_size=(2,2))(l7)
print(f"Layer 8 maxPooling => {l8.shape}")

l9 = Conv2D(64, (3, 3), padding="same")(l8)
print(f"Layer 9 conv2d => {l9.shape}")
l10 = MaxPooling2D(pool_size=(2,2))(l9)
print(f"Layer 10 maxPooling => {l10.shape}")

l11 = Conv2D(64, (3, 3), padding="same")(l10)
print(f"Layer 11 conv2d => {l11.shape}")
l12 = MaxPooling2D(pool_size=(2,2))(l11)
print(f"Layer 12 maxPooling => {l12.shape}")

l13 = Flatten()(l12)
print(f"Layer 13 Flatten => {l13.shape}")

l14 = Dense(512)(l13)
print(f"Layer 14 Dense => {l14.shape}")

# ltest = Reshape

l15 = RepeatVector(10)(l14)
print(f"Layer 15 Dense => {l15.shape}")

l16 = LSTM(10, return_sequences=True)(l15)
print(f"Layer 16 Dense => {l16.shape}")

l17 = TimeDistributed(Dense(len(chars)))(l16)
print(f"Layer 17 Dense => {l17.shape}")
