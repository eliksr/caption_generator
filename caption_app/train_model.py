import caption_generator
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import caption_generator


def train_model(weight = None, main_folder='../Xray_text/front', batch_size=1024, epochs = 10):

    cg = caption_generator.CaptionGenerator(main_folder)
    model = cg.create_model()

    if weight != None:
        model.load_weights(weight)

    counter = 0
    file_name = 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [tensor_board]
    model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=epochs, verbose=2, callbacks=callbacks_list)
    try:
        model.save(os.path.join(main_folder, 'WholeModel.h5'))
        model.save_weights(os.path.join(main_folder, 'Weights.h5'),overwrite=True)
    except:
        print ("Error in saving model.")
    print ("Training complete...\n")


if __name__ == '__main__':
    # main_dir = '../Xray_text/front_top_3'
    # train_model(main_folder=main_dir, epochs=20)

    weight = '/Users/esror/PycharmProjects/caption_generator/Xray_text/front_top_3/WholeModel.h5'
    test_image = '1013_IM-0013-1001.dcm'
    test_img_dir = '../Xray_text/xray_testImages.txt'
    eval_m = caption_generator.eval_model.Eval()
    print(eval_m('../Xray_text/front_top_3', weight, test_image))