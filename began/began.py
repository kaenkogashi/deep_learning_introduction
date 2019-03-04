import os

import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Activation, Flatten, Dense, UpSampling2D, Reshape, Lambda, Input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import img_to_array, array_to_img

def save_imgs(path, imgs, rows, cols):
    """画像をタイル状にならべて保存する
    
    Arguments:
        path (str): 保存先のファイルパス
        imgs (np.array): 保存する画像のリスト
        rows (int): タイルの縦のサイズ
        cols (int): タイルの横のサイズ
        
    Returns:
        None
    """
    base_width = imgs.shape[1]
    base_height = imgs.shape[2]
    channels = imgs.shape[3]
    output_shape = (
        base_height*rows,
        base_width*cols,
        channels
    )
    buffer = np.zeros(output_shape)
    for row in range(rows):
        for col in range(cols):
            img = imgs[row*cols + col]
            buffer[
                row*base_height:(row + 1)*base_height,
                col*base_width:(col + 1)*base_width
            ] = img
    array_to_img(buffer).save(path)


DATA_DIR = 'data/faces/'
BATCH_SIZE = 16
IMG_SHAPE = (64, 64, 3)

data_gen = ImageDataGenerator(rescale=1/255.)
train_data_generator = data_gen.flow_from_directory(
    directory=DATA_DIR,
    classes=['faces'],
    class_mode=None,
    batch_size=BATCH_SIZE,
    target_size=IMG_SHAPE[:2]
)

def build_encoder(input_shape, z_size, n_filters, n_layers):
    """Encoderを構築する
    
    Arguments:
        input_shape (int): 画像のshape
        z_size (int): 特徴空間の次元数
        n_filters (int): フィルタ数
        
    Returns:
        model (Model): Encoderモデル 
    """
    model = Sequential()
    model.add(
        Conv2D(
            n_filters,
            3,
            activation='elu',
            input_shape=input_shape,
            padding='same'
        )
    )
    model.add(Conv2D(n_filters, 3, padding='same'))
    for i in range(2, n_layers + 1):
        model.add(
            Conv2D(
                i*n_filters,
                3,
                activation='elu',
                padding='same'
            )
        )
        model.add(
                Conv2D(
                i*n_filters,
                3,
                activation='elu',
                strides=2,
                padding='same'
            )
        )
    model.add(Conv2D(n_layers*n_filters, 3, padding='same'))
    model.add(Flatten())
    model.add(Dense(z_size))
    
    return model


def build_decoder(output_shape, z_size, n_filters, n_layers):
    """Decoderを構築する
    
    Arguments:
        output_shape (np.array): 画像のshape
        z_size (int): 特徴空間の次元数
        n_filters (int): フィルタ数
        n_layers (int): レイヤー数数
        
    Returns:
        model (Model): Decoderモデル 
    """
    # UpSampling2Dで何倍に拡大されるか
    scale = 2**(n_layers - 1)
    # 最初の畳み込み層の入力サイズをscaleから逆算
    fc_shape = (
        output_shape[0]//scale,
        output_shape[1]//scale,
        n_filters
    )
    # 全結合層で必要なサイズを逆算
    fc_size = fc_shape[0]*fc_shape[1]*fc_shape[2]
    
    model = Sequential()
    # 全結合層
    model.add(Dense(fc_size, input_shape=(z_size,)))
    model.add(Reshape(fc_shape))
    
    # 畳み込み層の繰り返し
    for i in range(n_layers - 1):
        model.add(
            Conv2D(
                n_filters,
                3,
                activation='elu',
                padding='same'
            )
        )
        model.add(
            Conv2D(
                n_filters,
                3,
                activation='elu',
                padding='same'
            )
        )
        model.add(UpSampling2D())
        
    # 最後の層はUpSampling2Dが不要
    model.add(
        Conv2D(
            n_filters,
            3,
            activation='elu',
            padding='same'
        )
    )
    model.add(
        Conv2D(
            n_filters,
            3,
            activation='elu',
            padding='same'
        )
    )
    # 出力層で3チャンネルに
    model.add(Conv2D(3, 3, padding='same'))
    
    return model

def build_generator(img_shape, z_size, n_filters, n_layers):
    decoder = build_decoder(
        img_shape, z_size, n_filters, n_layers
    )
    return decoder

def build_discriminator(img_shape, z_size, n_filters, n_layers):
    encoder = build_encoder(
        img_shape, z_size, n_filters, n_layers
    )
    decoder = build_decoder(
        img_shape, z_size, n_filters, n_layers
    )
    return keras.models.Sequential((encoder, decoder))

def build_discriminator_trainer(discriminator):
    img_shape = discriminator.input_shape[1:]
    real_inputs = Input(img_shape)
    fake_inputs = Input(img_shape)
    real_outputs = discriminator(real_inputs)
    fake_outputs = discriminator(fake_inputs)

    return Model(
        inputs=[real_inputs, fake_inputs],
        outputs=[real_outputs, fake_outputs]
    )

n_filters = 64  #  フィルタ数
n_layers = 4 # レイヤー数
z_size = 32  #  特徴空間の次元

generator = build_generator(
    IMG_SHAPE, z_size, n_filters, n_layers
)
discriminator = build_discriminator(
    IMG_SHAPE, z_size, n_filters, n_layers
)
discriminator_trainer = build_discriminator_trainer(
    discriminator
)

generator.summary()
# discriminator.layers[1]が Decoder を表す
discriminator.layers[1].summary()

from tensorflow.python.keras.losses import mean_absolute_error


def build_generator_loss(discriminator):
    # discriminator を使って損失関数を定義
    def loss(y_true, y_pred):
        # y_true はダミー
        reconst = discriminator(y_pred)
        return mean_absolute_error(
            reconst,
            y_pred
        )
    return loss

# 初期の学習率(Generator)
g_lr = 0.0001

generator_loss = build_generator_loss(discriminator)
generator.compile(
    loss=generator_loss,
    optimizer=Adam(g_lr)
)

# 初期の学習率(Discriminator)
d_lr = 0.0001

# k_varは数値(普通の変数)
k_var = 0.0
# k はKeras(TensorFlow)のVariable
k = K.variable(k_var)
discriminator_trainer.compile(
    loss=[
        mean_absolute_error,
        mean_absolute_error
    ],
    loss_weights=[1., -k],
    optimizer=Adam(d_lr)
)

def measure(real_loss, fake_loss, gamma):
    return real_loss + np.abs(gamma*real_loss - fake_loss)


# kの更新に利用するパラメータ
GAMMA = 0.5
LR_K = 0.001

# 繰り返し数。100000〜1000000程度を指定
TOTAL_STEPS = 100000

# モデルや確認用の生成画像を保存するディレクトリ
MODEL_SAVE_DIR = 'began/models'
IMG_SAVE_DIR = 'began/imgs'
# 確認用に5x5個の画像を生成する
IMG_SAMPLE_SHAPE = (5, 5)
N_IMG_SAMPLES = np.prod(IMG_SAMPLE_SHAPE)


# 保存先がなければ作成
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# サンプル画像用のランダムシード
sample_seeds = np.random.uniform(
    -1, 1, (N_IMG_SAMPLES, z_size)
)

history = []
logs = []

for step, batch in enumerate(train_data_generator):
    # サンプル数がBATCH_SIZEに満たない場合はスキップ
    # 全体の画像枚数がBATCH_SIZEの倍数出ない場合に発生
    if len(batch) < BATCH_SIZE:
        continue
    
    # 学習終了
    if step > TOTAL_STEPS:
        break

    # ランダムな値を生成
    z_g = np.random.uniform(
        -1, 1, (BATCH_SIZE, z_size)
    )
    z_d = np.random.uniform(
        -1, 1, (BATCH_SIZE, z_size)
    )
    
    # 生成画像(discriminatorの学習に利用)
    g_pred = generator.predict(z_d)
    
    # generatorを1ステップ分学習させる
    generator.train_on_batch(z_g, batch)
    # discriminatorを1ステップ分学習させる
    _, real_loss, fake_loss = discriminator_trainer.train_on_batch(
            [batch, g_pred],
            [batch, g_pred]
    )

    # k を更新
    k_var += LR_K*(GAMMA*real_loss - fake_loss)
    K.set_value(k, k_var)
    

    # g_measure を計算するためにlossを保存
    history.append({
        'real_loss': real_loss,
        'fake_loss': fake_loss
    })

    # 1000回に1度ログを表示
    if step%1000 == 0:
        # 過去1000回分の measure を平均
        measurement = np.mean([
            measure(
                loss['real_loss'],
                loss['fake_loss'],
                GAMMA
            )
            for loss in history[-1000:]
        ])
        
        logs.append({
            'k': K.get_value(k),
            'measure': measurement,
            'real_loss': real_loss,
            'fake_loss': fake_loss
        })
        print(logs[-1])

        # 画像を保存  
        img_path = '{}/generated_{}.png'.format(
            IMG_SAVE_DIR,
            step
        )
        save_imgs(
            img_path,
            generator.predict(sample_seeds),
            rows=IMG_SAMPLE_SHAPE[0],
            cols=IMG_SAMPLE_SHAPE[1]
        )
        # 最新のモデルを保存
        generator.save('{}/generator_{}.hd5'.format(MODEL_SAVE_DIR, step))
        discriminator.save('{}/discriminator_{}.hd5'.format(MODEL_SAVE_DIR, step))