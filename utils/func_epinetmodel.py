"""
Updated for TensorFlow 2.x compatibility & Dynamic shape calculation
Original Author: shinyonsei2
"""

from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Input,
    Reshape,
    concatenate,
)
from keras.models import Model, Sequential
from keras.optimizers import RMSprop


def layer1_multistream(input_dim1, input_dim2, input_dim3, filt_num):
    """Multi-Stream layer : Conv - Relu - Conv - BN - Relu"""
    seq = Sequential()

    # 3 Blocks. Each block has 2 Conv2D with 'valid' padding.
    # Total reduction = 3 * 2 = 6 pixels.
    for i in range(3):
        if i == 0:
            seq.add(
                Conv2D(
                    int(filt_num),
                    (2, 2),
                    input_shape=(input_dim1, input_dim2, input_dim3),
                    padding="valid",
                    name="S1_c1%d" % (i),
                )
            )
        else:
            seq.add(
                Conv2D(
                    int(filt_num),
                    (2, 2),
                    padding="valid",
                    name="S1_c1%d" % (i),
                )
            )

        seq.add(Activation("relu", name="S1_relu1%d" % (i)))
        seq.add(Conv2D(int(filt_num), (2, 2), padding="valid", name="S1_c2%d" % (i)))
        seq.add(BatchNormalization(axis=-1, name="S1_BN%d" % (i)))
        seq.add(Activation("relu", name="S1_relu2%d" % (i)))

    seq.add(Reshape((input_dim1 - 6, input_dim2 - 6, int(filt_num))))

    return seq


def layer2_merged(input_dim1, input_dim2, input_dim3, filt_num, conv_depth):
    """Merged layer : Conv - Relu - Conv - BN - Relu"""
    seq = Sequential()

    for i in range(conv_depth):
        if i == 0:
            seq.add(
                Conv2D(
                    filt_num,
                    (2, 2),
                    padding="valid",
                    input_shape=(input_dim1, input_dim2, input_dim3),
                    name="S2_c1%d" % (i),
                )
            )
        else:
            seq.add(
                Conv2D(
                    filt_num,
                    (2, 2),
                    padding="valid",
                    name="S2_c1%d" % (i),
                )
            )

        seq.add(Activation("relu", name="S2_relu1%d" % (i)))
        seq.add(Conv2D(filt_num, (2, 2), padding="valid", name="S2_c2%d" % (i)))
        seq.add(BatchNormalization(axis=-1, name="S2_BN%d" % (i)))
        seq.add(Activation("relu", name="S2_relu2%d" % (i)))

    return seq


def layer3_last(input_dim1, input_dim2, input_dim3, filt_num):
    """last layer : Conv - Relu - Conv"""
    seq = Sequential()

    for i in range(1):
        seq.add(
            Conv2D(
                filt_num,
                (2, 2),
                padding="valid",
                input_shape=(input_dim1, input_dim2, input_dim3),
                name="S3_c1%d" % (i),
            )
        )
        seq.add(Activation("relu", name="S3_relu1%d" % (i)))

    seq.add(Conv2D(1, (2, 2), padding="valid", name="S3_last"))

    return seq


def define_epinet(sz_input_h, sz_input_w, view_n, conv_depth, filt_num, learning_rate):
    """4-Input : Conv - Relu - Conv - BN - Relu"""
    input_stack_90d = Input(
        shape=(sz_input_h, sz_input_w, len(view_n)), name="input_stack_90d"
    )
    input_stack_0d = Input(
        shape=(sz_input_h, sz_input_w, len(view_n)), name="input_stack_0d"
    )
    input_stack_45d = Input(
        shape=(sz_input_h, sz_input_w, len(view_n)), name="input_stack_45d"
    )
    input_stack_M45d = Input(
        shape=(sz_input_h, sz_input_w, len(view_n)), name="input_stack_M45d"
    )

    """ 4-Stream layer : Conv - Relu - Conv - BN - Relu """
    mid_90d = layer1_multistream(sz_input_h, sz_input_w, len(view_n), int(filt_num))(
        input_stack_90d
    )
    mid_0d = layer1_multistream(sz_input_h, sz_input_w, len(view_n), int(filt_num))(
        input_stack_0d
    )
    mid_45d = layer1_multistream(sz_input_h, sz_input_w, len(view_n), int(filt_num))(
        input_stack_45d
    )
    mid_M45d = layer1_multistream(sz_input_h, sz_input_w, len(view_n), int(filt_num))(
        input_stack_M45d
    )

    """ Merge layers """
    mid_merged = concatenate([mid_90d, mid_0d, mid_45d, mid_M45d], name="mid_merged")

    """ Merged layer : Conv - Relu - Conv - BN - Relu """
    layer2_input_h = sz_input_h - 6
    layer2_input_w = sz_input_w - 6

    mid_merged_ = layer2_merged(
        layer2_input_h,
        layer2_input_w,
        int(4 * filt_num),
        int(4 * filt_num),
        conv_depth,
    )(mid_merged)

    """ Last Conv layer : Conv - Relu - Conv """
    reduction_layer1 = 6
    reduction_layer2 = conv_depth * 2

    layer3_input_h = sz_input_h - reduction_layer1 - reduction_layer2
    layer3_input_w = sz_input_w - reduction_layer1 - reduction_layer2

    output = layer3_last(
        layer3_input_h, layer3_input_w, int(4 * filt_num), int(4 * filt_num)
    )(mid_merged_)

    model_512 = Model(
        inputs=[input_stack_90d, input_stack_0d, input_stack_45d, input_stack_M45d],
        outputs=[output],
    )
    opt = RMSprop(learning_rate=learning_rate)
    model_512.compile(optimizer=opt, loss="mae")
    model_512.summary()

    return model_512
