import os
import cv2
import numpy as np




#-------------PREPROCESS-----------------------------------------------------------

def preprocess_data(video_dir, label, frame_count=10, frame_size=(256,256)):
    videos=[]
    labels=[]
    i = 0
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        frames=[]
        
        i += 1
        print(i, end=" ")
        if i == 650:
            break
        
        while len(frames) < frame_count:
            ret, frame = cap.read()
           
            if not ret:
                break
        
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
            
        cap.release()
        
        if len(frames) == frame_count:
            videos.append(np.array(frames))
            labels.append(label)
    
    
    return videos, labels




real_videos_dir = 'data/Celeb-real'
fake_videos_dir = 'data/Celeb-synthesis'


real_videos, real_labels = preprocess_data(real_videos_dir, 1)
fake_videos, fake_labels = preprocess_data(fake_videos_dir, 0)


videos = real_videos + fake_videos
labels = real_labels + fake_labels

videos = np.array(videos, dtype=np.uint8)  # Daha az RAM kullanır
labels = np.array(labels, dtype=np.int8)   # Etiketler için küçük boyutlu tip

print(f'Shape of video dataset: {videos.shape}')
print(f'Shape of labels dataset: {labels.shape}')





#----------------------------------CNN-----------------------------------------------

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


class DeepFakeDetectionModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()
        
        
    
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        
        
        #CONV1
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', name='conv3d_1')(inputs)
        x = BatchNormalization(name='batch_normalization_1')(x)
        x = MaxPooling3D((2,2,2), name='max_pooling3d_1')(x)
        
        
        #CONV2
        x = Conv3D(16, (3, 3, 3), activation='relu', padding='same', name='conv3d_2')(x)
        x = BatchNormalization(name='batch_normalization_2')(x)
        x = MaxPooling3D((2,2,2), name='max_pooling3d_2')(x)
        
        #CONV3
        x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='conv3d_3')(x)
        x = BatchNormalization(name='batch_normalization_3')(x)
        x = MaxPooling3D((2,2,2), name='max_pooling3d_3')(x)
        
        
        
        #FLAT
        x = Flatten(name='flatten')(x)
        
        #DENSE
        x = Dense(64, activation='relu', name='dense_1')(x)
        
        #DROPOUT
        x = Dropout(0.5, name='dropout')(x)
        
        #OUTPUT
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        
        #COMPILE
        
        model = Model(inputs, outputs, name='DeepFakeDetectionModel')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    
    
    
    # def summary(self):
    #     return self.model.summary()
    
    
    
    def train(self, train_data, validation_data, epochs=150, batch_size=16):
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        return history
    
    
    def evaluate(self, test_data):
        return self.model.evaluate(test_data)
    
    
    def predict(self, data):
        return self.model.predict(data)
    
    def save(self, filepath):
        self.model.save(filepath)
        
        
    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        
        
    
    
#--------------------TRAIN------------------------
# frame_count = 10
# height = width = 256
# channels = 3
# input_shape = (frame_count, height, width, channels)  # (10, 256, 256, 3)
        
        
input_shape = (10, 256, 256, 3)

deepfake_detector = DeepFakeDetectionModel(input_shape)


from sklearn.model_selection import train_test_split

train_videos, val_videos, train_labels, val_labels = train_test_split(videos, labels, test_size=0.2, random_state=42)




train_data = tf.data.Dataset.from_tensor_slices((train_videos, train_labels)).batch(8)
validation_data = tf.data.Dataset.from_tensor_slices((val_videos, val_labels)).batch(8)

deepfake_detector.train(train_data, validation_data, epochs=28)




deepfake_detector.save('deepfake_detection_model.h5')

#----------------------------PREDICT PREPROCESS-------------------------

def preprocess_frame(frame):
    img = cv2.resize(frame, (256, 256))
    img = img.astype('float32') / 255.0
    return img


def detect_deepfake(video_path, frame_count=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    deepfake_scores = []
    i=0
    while True  :
        i += 1
        if i==21 :
            break
        ret, frame = cap.read()
        if not ret:
            break
        
     
        img = preprocess_frame(frame)
        
        
        frames.append(img)
        
        
        if len(frames) == frame_count:
            
            frames_array = np.array(frames)
            
            
            prediction = deepfake_detector.predict(np.expand_dims(frames_array, axis=0))  
            deepfake_scores.append(prediction[0][0])
            
            frames.pop(0)

    cap.release()
    
    
    avg_score = np.mean(deepfake_scores) if deepfake_scores else 0
    print(f"Average Deepfake Score: {avg_score}")

    
    if avg_score > 0.5:
        print("The video is likely a Deepfake.")
    else:
        print("The video is likely Real.")


# video_path = 'data/Celeb-synthesis/id0_id2_0007.mp4'TEST VIDEO

detect_deepfake(video_path)











