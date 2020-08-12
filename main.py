from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os
import numpy as np
import pickle
from keras.models import load_model

app = Flask(__name__)

#環境変数取得
YOUR_CHANNEL_ACCESS_TOKEN = os.environ["YOUR_CHANNEL_ACCESS_TOKEN"]
YOUR_CHANNEL_SECRET = os.environ["YOUR_CHANNEL_SECRET"]

line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)

# かなリスト・辞書・モデル
with open('kana_chars.pickle', mode='rb') as f:
    chars_list = pickle.load(f)
with open('char_indices.pickle', mode='rb') as f:
    char_indices = pickle.load(f)
with open('indices_char.pickle', mode='rb') as f:
    indices_char = pickle.load(f)

n_char = len(chars_list)
max_length_x = 128

encoder_model = None
decoder_model = None


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# テキストメッセージが送信されたときの処理
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # 500エラー対策
    if event.reply_token == "00000000000000000000000000000000":
        return
    text = event.message.text

    if not is_invalid(text):
        response = respond(text)
        reply_message(event, response)
    else:
        response = "ごめんなさい、ひらがな・カタカナしかつかえません！"
        reply_message(event, response)

### ここから賢治bot用の関数 ###
# 使える文字かどうか判定する関数
def is_invalid(message):
    is_invalid =False
    global chars_list
    for char in message:
        if char not in chars_list:
            is_invalid = True
    return is_invalid

# 文章をone-hot表現に変換する関数
def sentence_to_vector(sentence):
    global chars_list, char_indices, n_char, max_length_x
    vector = np.zeros((1, max_length_x, n_char), dtype=np.bool)
    for j, char in enumerate(sentence):
        vector[0][j][char_indices[char]] = 1
    return vector

# 文章を生成する関数
def respond(message, beta=5):
    global encoder_model, decoder_model, n_char, indices_char
    # 一番初めだけモデルをロード
    if encoder_model is None:
        encoder_model = load_model('encoder_model.h5')
    if decoder_model is None:
        decoder_model = load_model('decoder_model.h5')

    vec = sentence_to_vector(message)  # 入力した文字列をone-hot表現に変換
    state_value = encoder_model.predict(vec)  # エンコーダーへ入力
    y_decoder = np.zeros((1, 1, n_char))  # decoderの出力を格納する配列
    y_decoder[0][0][char_indices['\t']] = 1  # decoderの最初の入力はタブ。one-hot表現にする。

    respond_sentence = ""  # 返答の文字列
    while True:
        y, h = decoder_model.predict([y_decoder, state_value])
        p_power = y[0][0] ** beta  # 確率分布の調整
        next_index = np.random.choice(len(p_power), p=p_power / np.sum(p_power))
        next_char = indices_char[next_index]  # 次の文字

        if (next_char == "\n" or len(respond_sentence) >= max_length_x):
            break  # 次の文字が改行のとき、もしくは最大文字数を超えたときは終了

        respond_sentence += next_char
        y_decoder = np.zeros((1, 1, n_char))  # 次の時刻の入力
        y_decoder[0][0][next_index] = 1

        state_value = h  # 次の時刻の状態

    return respond_sentence

# 文章を受け取って返答する関数
def reply_message(event, messages):
    line_bot_api.reply_message(
        event.reply_token,
        messages=messages,
    )

if __name__ == "__main__":
#    app.run()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)