import requests
from bs4 import BeautifulSoup

def get_github_profile(username):
    url = f"https://github.com/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def main():
    username = input("Enter the GitHub username: ")
    html_content = get_github_profile(username)
    if html_content:
        # 解析HTML内容
        soup = BeautifulSoup(html_content, 'html.parser')
        # 这里可以添加更多的解析逻辑
        print(soup.prettify())  # 打印整个页面的HTML内容
    else:
        print("Failed to retrieve the profile.")

if __name__ == "__main__":
    main()
