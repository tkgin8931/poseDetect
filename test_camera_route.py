import requests

# Flaskサーバーがローカルで5000番ポートで起動している前提
url = 'http://127.0.0.1:5000/camera'

response = requests.get(url, stream=True)
print('Status code:', response.status_code)
print('Content-Type:', response.headers.get('Content-Type'))

# 最初の画像フレームを保存して確認
with open('test_frame.jpg', 'wb') as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)
            break  # 最初のフレームだけ保存
print('最初のフレームを test_frame.jpg に保存しました')
