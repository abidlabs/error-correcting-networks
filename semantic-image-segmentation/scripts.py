import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

from keras.layers import Input, Embedding, LSTM, Dense, Conv2D, Flatten, Reshape, Add, Concatenate, MaxPool2D, Lambda
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras import backend as K

from model.unet import dice_coef
from dataset_parser import generator_coarse, generator, generator_miscl


def np_normalized_iou_coef(y_true, y_pred, n_classes=3, has_batch_axis=False):
    mean_iou = 0
    for i in range(n_classes):
        if has_batch_axis:
            true = y_true[:, :, :, i]
            pred = y_pred[:, :, :, i]
        else:
            true = y_true[:, :, i]
            pred = y_pred[:, :, i]            
        tp = np.sum(true * pred)
        fp = np.sum((1-true) * pred)
        fn = np.sum((1-pred) * true)
        iou = tp / (tp + fp + fn)
        print(i)
        print(tp, fp, fn)        
        print(iou)
        mean_iou += iou
    print('')
    return mean_iou / n_classes


def np_hard_normalized_hard_iou_coef(y_true, y_pred, n_classes=3):
    mean_iou = 0
    for i in range(n_classes):
        pred = np.argmax(y_pred, axis=-1) == i
        pred = pred.flatten()
        true = np.argmax(y_true, axis=-1) == i
        true = true.flatten()
        tp = np.sum(true * pred)
        fp = np.sum((1-true) * pred)
        fn = np.sum((1-pred) * true)
        iou = tp / (tp + fp + fn)
        mean_iou += iou
    return mean_iou / n_classes


def view_paired_predictions_on_training_data(model, n=5, ignore_pedestrians=False, coarse_or_miscl='coarse', bad_generator=None):
    if coarse_or_miscl=='coarse':
        coarse_generator = generator_coarse.data_generator('dataset_parser/data-coarse.h5', 1, 'train', ignore_pedestrians=ignore_pedestrians)
    elif coarse_or_miscl=='miscl':
        coarse_generator = generator_miscl.data_generator('dataset_parser/data.h5', 1, 'train', ignore_pedestrians=ignore_pedestrians, misclassified_frac=0.5)
    else:
        raise ValueError('invalid choice for param: coarse_or_miscl')
        
    fine_generator = generator.data_generator('dataset_parser/data.h5', 1, 'train', ignore_pedestrians=ignore_pedestrians)

    for i in range(n):
        plt.figure(figsize=[12, 8])
        x, y = next(coarse_generator)
        x_, y_ = next(fine_generator)

        orig_shaded_image = (x[0] + 1)/2
        plt.subplot(2, 2, 1+0)
        plt.imshow(orig_shaded_image)
        plt.title('Image {}'.format(i))    

        label_shaded_image = np.argmax(y[0], axis=-1)
        plt.subplot(2, 2, 1+1)
        plt.imshow(label_shaded_image)
        plt.title('{} Image {} label'.format(coarse_or_miscl, i))

        label_shaded_pred = np.argmax(y_[0], axis=-1)        
        plt.subplot(2, 2, 3)
        plt.imshow(label_shaded_pred)    
        plt.title('Fine Image {} Label'.format(i))
        
        y_pred_raw = model.predict(x)[0]
        y_pred = np.argmax(y_pred_raw, axis=-1)
        
        y_pred_hard = np.zeros_like(y_pred_raw)
        for r in range(y_pred_raw.shape[0]):
            y_pred_hard[r, np.arange(y_pred_raw.shape[1]), y_pred_raw[r].argmax(axis=-1)] = 1
        score_raw = np_normalized_iou_coef(y_[0], y_pred_raw)
        score_raw = np.round(score_raw, 3)
        score = np_normalized_iou_coef(y_[0], y_pred_hard)
        score = np.round(score, 3)
        score_2 = np_hard_normalized_hard_iou_coef(y_[0], y_pred_raw)
        score_2 = np.round(score_2, 3)
        
        plt.subplot(2, 2, 4)
        plt.imshow(y_pred)
        plt.title('Model Prediction {}, Score:{}, Score2:{}, ScoreRaw: {}'.format(i, score, score_2, score_raw))   

        plt.suptitle('Figure {}'.format(i))
        plt.tight_layout()
        
def view_corrections_on_training_data(coarse_model, ec_model, n=1, ignore_pedestrians=False, input_window_size=60, output_window_size=2, diff=True, use_model_pred=True, use_x=True, coarse_or_miscl='coarse'):
    
    if coarse_or_miscl == 'coarse':
        coarse_generator = generator_coarse.data_generator('dataset_parser/data-coarse.h5', 1, 'train', ignore_pedestrians=ignore_pedestrians)
    elif coarse_or_miscl=='miscl':
        coarse_generator = generator_miscl.data_generator('dataset_parser/data.h5', 1, 'train', ignore_pedestrians=ignore_pedestrians, misclassified_frac=1)
    else:
        raise ValueError('invalid choice for param: coarse_or_miscl')
        
        
    fine_generator = generator.data_generator('dataset_parser/data.h5', 1, 'train', ignore_pedestrians=ignore_pedestrians)
    skip = output_window_size
    
    for i in range(n):
        plt.figure(figsize=[12, 12])
        x, y_coarse = next(coarse_generator)
        _, y_ = next(fine_generator)

        orig_shaded_image = (x[0] + 1)/2
        plt.subplot(3, 2, 1+0)
        plt.imshow(orig_shaded_image)
        plt.title('Coarse Image {}'.format(i))    

        label_shaded_image = np.argmax(y_coarse[0], axis=-1)
        plt.subplot(3, 2, 1+1)
        plt.imshow(label_shaded_image)
        plt.title('Coarse Image {} label'.format(i))

        label_shaded_pred = np.argmax(y_[0], axis=-1)        
        plt.subplot(3, 2, 3)
        plt.imshow(label_shaded_pred)    
        plt.title('Fine Image {} Label'.format(i))
        
        y_pred_raw = coarse_model.predict(x)
        y_pred = np.argmax(y_pred_raw[0], axis=-1)
        plt.subplot(3, 2, 4)
        plt.imshow(y_pred)
        plt.title('Model Prediction for Image {}'.format(i))   
        
        if use_model_pred:
            y_recon = np.copy(y_pred_raw)
        else:
            y_recon = np.copy(y_coarse)
            
        inp_x = list()
        inp_y = list()
        output_offset = (input_window_size - output_window_size) // 2

        for r in range(0, x.shape[1] - input_window_size + 1, skip):
            for c in range(0, x.shape[2] - input_window_size + 1, skip):
                inp_x.append(x[:, r: r+input_window_size, c: c+input_window_size, :])
                if use_model_pred:
                    inp_y.append(y_pred_raw[:, r: r+input_window_size, c: c+input_window_size, :])
                else:
                    inp_y.append(y_coarse[:, r: r+input_window_size, c: c+input_window_size, :])

        inp_x = np.concatenate(inp_x, axis=0)
        inp_y = np.concatenate(inp_y, axis=0)
        if use_x:
            out = ec_model.predict([inp_x, inp_y])
        else:
            out = ec_model.predict(inp_y)
        
        index = 0
        for r in range(0, x.shape[1] - input_window_size + 1, skip):
            for c in range(0, x.shape[2] - input_window_size + 1, skip):
                y_recon[:, r+output_offset: r+output_offset+output_window_size,
                        c+output_offset: c+output_offset+output_window_size, :] = out[index: index+1]
                index += 1
                     
        y_recon = np.argmax(y_recon[0], axis=-1)
        
        plt.subplot(3, 2, 5)
        plt.imshow(y_recon)
        plt.title('Reconstructed Label for Image {}'.format(i))           
        
        plt.suptitle('Figure {}'.format(i))
        plt.tight_layout()

        
def view_predictions_on_validation_data(model, n=5):
    val_generator = generator.data_generator('dataset_parser/data.h5', 1, 'val', ignore_pedestrians=True)

    for i in range(n):
        plt.figure(figsize=[12, 8])
        x, y = next(val_generator)

        orig_shaded_image = (x[0] + 1)/2
        plt.subplot(2, 2, 1+0)
        plt.imshow(orig_shaded_image)
        plt.title('Val Image {}'.format(i))    

        label_shaded_image = np.argmax(y[0], axis=-1)
        plt.subplot(2, 2, 1+1)
        plt.imshow(label_shaded_image)
        plt.title('Fine Image {} labels'.format(i))

        y_pred = np.argmax(model.predict(x)[0], axis=-1)
        plt.subplot(2, 2, 4)
        plt.imshow(y_pred)
        plt.title('Model Prediction'.format(i))   

        plt.suptitle('Figure {}'.format(i))
        plt.tight_layout()        

def get_ec_model(input_window_size, output_window_size, dice_coef=False):
    assert output_window_size == input_window_size // 8, "For now, output_window_size must be input_window_size // 8"
    
    lr_init = 1e-3
    lr_decay = 5e-4
    
    inp_y = Input(shape=(input_window_size, input_window_size, 3))
    inp_x = Input(shape=(input_window_size, input_window_size, 3))
    h = Concatenate()([inp_y, inp_x])
    h = Conv2D(8, (4, 4), padding='same', activation='relu')(h)
    h = MaxPool2D((2, 2), padding='same')(h)
    h = Conv2D(8, (4, 4), padding='same', activation='relu')(h)
    h = MaxPool2D((2, 2), padding='same')(h)
    h = Conv2D(16, (4, 4), padding='same', activation='relu')(h)
    h = MaxPool2D((2, 2), padding='same')(h)
    h = Conv2D(16, (4, 4), padding='same', activation='relu')(h)
    h = Dense(3, activation='softmax')(h)
    ec_model = Model(inputs=[inp_y, inp_x], outputs=h)

    ec_model.summary()
    metrics=['mean_squared_error']
    if dice_coef:
        metrics.append(dice_coef)
    ec_model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                     loss='categorical_crossentropy', metrics=metrics)    
    
    return ec_model


def get_identity_model():
    lr_init = 1e-3
    lr_decay = 5e-4
    
    model = Sequential()
    model.add(Lambda(lambda x: x))
#     model.summary()
    metrics= ['mean_squared_error', dice_coef]
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy', metrics=metrics)    
    return model


def get_ec_batch(x, y_fine, y_coarse, input_window_size, output_window_size, skip, diff, balanced=False):
    inp_x = list()
    inp_y = list()
    out = list()
    output_offset = (input_window_size - output_window_size) // 2
    balance_counter = 10

    for r in range(0, x.shape[1] - input_window_size + 1, skip):
        for c in range(0, x.shape[2] - input_window_size + 1, skip):
            if not(balanced):
                inp_x.append(x[:, r: r+input_window_size, c: c+input_window_size, :])
                inp_y.append(y_coarse[:, r: r+input_window_size, c: c+input_window_size, :])
                out.append(y_fine[:, r+output_offset: r+output_offset+output_window_size, 
                                  c+output_offset: c+output_offset+output_window_size, :])
            else:
                assert output_window_size == 1, "balanced only works if output window=1"
                for j in range(len(y_fine)):
                    y_coarse_pix = y_coarse[j, r+output_offset, c+output_offset, :].argmax()
                    y_fine_pix = y_fine[j, r+output_offset, c+output_offset, :].argmax()
                    if y_fine_pix == y_coarse_pix and balance_counter > 0:
#                         print('-', end='')
                        inp_x.append(x[j:j+1, r: r+input_window_size, c: c+input_window_size, :])
                        inp_y.append(y_coarse[j:j+1, r: r+input_window_size, c: c+input_window_size, :])
                        out.append(y_fine[j:j+1, r+output_offset: r+output_offset+output_window_size, 
                                          c+output_offset: c+output_offset+output_window_size, :])
                        
                        balance_counter -= 1
                    elif not(y_fine_pix == y_coarse_pix):
#                         print('+', end='')
                        inp_x.append(x[j:j+1, r: r+input_window_size, c: c+input_window_size, :])
                        inp_y.append(y_coarse[j:j+1, r: r+input_window_size, c: c+input_window_size, :])
                        out.append(y_fine[j:j+1, r+output_offset: r+output_offset+output_window_size, 
                                          c+output_offset: c+output_offset+output_window_size, :])
                        
                        balance_counter += 3                   
                    else:
                        pass
                        
    
    inp_x = np.concatenate(inp_x, axis=0)
    inp_y = np.concatenate(inp_y, axis=0)
    out = np.concatenate(out, axis=0)
    return inp_x, inp_y, out


def test_get_ec_batch():
    shape = (8, 256, 512, 3)
    input_window_size = 60
    output_window_size = 10
    skip = 10
    
    def test1():  # correct shape
        x = np.random.random(shape)
        y = np.random.random(shape)
        y_ = np.random.random(shape)   

        xi, yi, out = get_ec_batch(x, y, y_, 
                                   input_window_size=input_window_size,
                                   output_window_size=output_window_size, 
                                   skip=skip, 
                                   diff=False)

        
        assert xi.shape == (7360, 60, 60, 3), "xi.shape = {}".format(xi.shape)
        assert yi.shape == xi.shape
        assert out.shape == (7360, 10, 10, 3)
        print('Test 1 passed')
        
    def test2():  # all the same
        x = np.random.random(shape)

        xi, yi, out = get_ec_batch(x, x, x, 
                                   input_window_size=input_window_size,
                                   output_window_size=output_window_size, 
                                   skip=skip, 
                                   diff=False)
        
        assert np.allclose(xi, yi)
        print('Test 2 passed')
    
    def test3():  # nothing skipped
        x = np.random.random(shape)
        y = np.random.random(shape)
        y_ = np.random.random(shape)   
        
        xi, yi, out = get_ec_batch(x, y, y_, 
                                   input_window_size=32,
                                   output_window_size=8, 
                                   skip=32, 
                                   diff=False)
        
        assert xi.size == x.size, "{} != {}".format(xi.size, x.size)
        assert yi.size == y.size, "{} != {}".format(yi.size, y.size)
        print('Test 3 passed')
    
    test1(); test2(); test3();
    

def compute_dice_on_ec_batches(y_val, y_val_pred_batch, input_window_size, output_window_size, skip=3, diff=True):
    y_shape = y_val.shape
    y_recon = np.copy(y_val)
    index = 0
    output_offset = input_window_size - output_window_size // 2
    
    for r in range(0, y_shape[1] - input_window_size, skip):
        for c in range(0, y_shape[2] - input_window_size, skip):
            for j in range(y_shape[0]):
                w = min(r+output_offset+output_window_size, y_shape[1]) - (r+output_offset)
                h = min(c+output_offset+output_window_size, y_shape[2]) - (c+output_offset)
                if diff:
                    y_diff = y_val_pred_batch[index, :w, :h, :] * 2 - 1  # To get back between -1 and 1
                    y_diff = y_recon[0, r+output_offset: r+output_offset+output_window_size, 
                            c+output_offset: c+output_offset+output_window_size, :] - y_diff
                    y_recon[0, r+output_offset: r+output_offset+output_window_size, 
                            c+output_offset: c+output_offset+output_window_size, :] = y_diff                    
                else:
                    y_recon[j, r+output_offset: r+output_offset+output_window_size, 
                            c+output_offset: c+output_offset+output_window_size, :] = y_val_pred_batch[index, :w, :h, :]
                index += 1
    
    return (2. * np.sum(y_val * y_recon) + 1.) / (np.sum(y_val) + np.sum(y_recon) + 1.)

def apply_ec_model(ec_model, x_batch, y_batch_dirty, input_window_size, output_window_size, skip):
        inp_x = list()
        inp_y = list()
    
        for r in range(0, y_batch_dirty.shape[1] - input_window_size, skip):
            for c in range(0, y_batch_dirty.shape[2] - input_window_size, skip):
                for j in range(x_batch.shape[0]):
                    inp_x.append(x_batch[j, r: r+input_window_size, c: c+input_window_size, :])
                    inp_y.append(y_batch_dirty[j, r: r+input_window_size, c: c+input_window_size, :])
        
        inp_x = np.stack(inp_x)
        inp_y = np.stack(inp_y)
        out = ec_model.predict([inp_x, inp_y])
        y_recon = np.copy(y_batch_dirty)
        
        index = 0
        output_offset = input_window_size - output_window_size // 2
        
        for r in range(0, y_batch_dirty.shape[1] - input_window_size, skip):
            for c in range(0, y_batch_dirty.shape[2] - input_window_size, skip):
                for j in range(x_batch.shape[0]):
                    w = min(r+output_offset+output_window_size, y_batch_dirty.shape[1]) - (r+output_offset)
                    h = min(c+output_offset+output_window_size, y_batch_dirty.shape[2]) - (c+output_offset)
                    y_recon[j, r+output_offset: r+output_offset+output_window_size, 
                            c+output_offset: c+output_offset+output_window_size, :] = out[index, :w, :h, :]
                    index += 1
                    
        return y_recon
