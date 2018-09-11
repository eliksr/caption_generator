import caption_app
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
import os
import caption_app.caption_generator as gen
import caption_app.eval_model as Val
import pickle
import traceback
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


def train_model(weight = None, main_folder='../Xray_text/front', batch_size=512, epochs = 10):

    cg = gen.CaptionGenerator(main_folder)
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
    except Exception as e:
        print ("Error in saving model.")
        traceback.print_exc()
    print ("Training complete...\n")

    model = load_model(os.path.join(main_folder, 'WholeModel.h5'))
    return model


if __name__ == '__main__':
    main_dir = '../Xray_text/side_top_3'
    nn_model = train_model(main_folder=main_dir, epochs=1000)
    # nn_model = load_model(os.path.join(main_dir, 'WholeModel.h5'))

    # weight = '/Users/esror/PycharmProjects/caption_app/Xray_text/front_top_5/WholeModel_1.h5'
    eval_m = Val.Eval()

    with open(os.path.join(main_dir, 'xray_train_dataset.txt'), 'r') as f:
        lines = f.read().splitlines()
        # test_image = '1020_IM-0017-1001.dcm0'
        sum_score = []
        for line in lines:
            if 'image_id' in line:
                continue
            items = line.split('<start>')
            if items[0][-2] == '0':
                hypo = eval_m.test_model('../Xray_text/side_top_3', nn_model, items[0][0:-1])
                ref = items[1][0:-6]
                pad_hypo = hypo.split(" ")
                # N = len(ref.strip().split(' ')) - len(hypo.strip().split(' '))
                # for i in range(N):
                #     pad_hypo.append("empty")
                # ref_set = set(ref.strip().split())
                # hypo_set = set(pad_hypo)
                score = sentence_bleu([ref.strip().split()], pad_hypo, weights=(1, 0, 0, 0))
                # score = len(ref_set.intersection(hypo_set))
                sum_score.append(score)
                # print('ref: {0} , hypo: {1} , score = {2}'.format(ref, hypo, score))

        print("Avg score is {0}".format(np.mean(sum_score)))


    # encoded_images = pickle.load(open(os.path.join(main_dir, "encoded_images.p"), "rb"))
    # print(len(encoded_images))