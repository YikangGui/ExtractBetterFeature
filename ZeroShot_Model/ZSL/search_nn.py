from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from wheel import Generator_searchNN
from sklearn.model_selection import train_test_split
import keras
from sklearn import svm

input_dimention = 1024
num_class = 65
batch_size = 64
epochs = 100
feat_fp = '../../data/searchLabel/ImageFeature_A1/'
fc_vector_test_fp = '../../data/list_tc/feat/CNN16/test_D_feat.npy'
test_image_labled_fp = '../../data/list_tc/imageName_lable/test_imageName_Lable_D.txt'
# read data
feat, label, test_feat, test_label = Generator_searchNN.readFeatAndLabel(
                                     feat_fp, fc_vector_test_fp,
                                     test_image_labled_fp)

x_train, x_val, y_train, y_val = train_test_split(feat, label, random_state=100)


'''
y_train = keras.utils.to_categorical(y_train, num_class)
y_val = keras.utils.to_categorical(y_val, num_class)
test_label = keras.utils.to_categorical(test_label, num_class)
model = Sequential()
model.add(Dense(num_class, activation='softmax', input_shape=(input_dimention,)))
# model.add(Dropout(0.5))
# model.add(Dense(num_class, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(test_feat, test_label))
score = model.evaluate(test_feat, test_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''

clf = svm.SVC(C=100, kernel='linear')
clf.fit(x_train, y_train)
score = clf.score(test_feat, test_label)
print('Test loss:', score)