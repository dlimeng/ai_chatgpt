import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

API_URL = 'https://api.poe.quora.com'

def get_user(user_id, access_token):
    endpoint = f'{API_URL}/users/{user_id}'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(endpoint, headers=headers)
    return response.json()

def get_unread_messages(user_id, access_token):
    endpoint = f'{API_URL}/users/{user_id}/messages/unread'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(endpoint, headers=headers)
    return response.json()

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()

    if 'message' in data:
        message = data['message']
        access_token = 'your_access_token'  # 替换为您的实际访问令牌

        user_id = data.get('userId')
        if user_id:
            user = get_user(user_id, access_token)
            print(f'User: {user}')

            unread_messages = get_unread_messages(user_id, access_token)
            # 在此处处理未读消息，例如：发送给其他用户或存储到数据库等
            print(f'Unread messages: {unread_messages}')

    return jsonify(status='success')

if __name__ == '__main__':
    app.run(debug=True)