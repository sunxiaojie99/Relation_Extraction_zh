from flask import Flask, jsonify, make_response, request, abort
import json
import time
from pprint import pprint

from complete_scripts.run import deal_one_sent
from relation_extraction.predict import predict_without_output_fixed_subj
from relation_extraction.hparams import hparams

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# http://127.0.0.1:5000/re/api/v1.0/predict
@app.route('/re/api/v1.0/predict', methods=['GET'])
def index():
    try:
        data = json.loads(request.data)  # 接收参数
        sentence = data['text']
        subject = data['subject']
    except Exception as e:
        print(e)
        abort(400)

    if subject not in sentence:
        abort(400)

    localtime = time.asctime(time.localtime(time.time()))
    print("获取object开始时间 :", localtime)

    sents = deal_one_sent(sentence=sentence, subject=subject)

    localtime = time.asctime(time.localtime(time.time()))
    print("获取object结束时间 :", localtime)

    hparams.device = 'cpu'
    hparams.is_add_entity_type = True
    res = predict_without_output_fixed_subj(hparams, sents)

    localtime = time.asctime(time.localtime(time.time()))
    print("预测结束时间 :", localtime)

    print(json.dumps(res, indent=4, ensure_ascii=False))
    return jsonify({'status': 'ok', 'res': res, 'inst_ori': data})


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'wrong input format'}), 400)


if __name__ == '__main__':
    app.run(port=5100, host="0.0.0.0", debug=True)
