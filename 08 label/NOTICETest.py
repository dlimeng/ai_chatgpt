import requests

NOTICE = 'NOTICE'

dependencies = []
with open('LICENSE.txt') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#'):
            continue

        fields = line.split(' - ')
        name = fields[0].strip()
        license = fields[-1].strip()
        version = ' - '.join(fields[1:-1])

        dependencies.append({
            'name': name,
            'version': version,
            'license': license
        })

with open(NOTICE, 'w') as f:
    for dependency in dependencies:
        f.write(f'=======================================\n')
        f.write(f'{dependency["name"]}  {dependency["version"]}\n')
        f.write(f'=======================================\n')
        f.write(f'{dependency["license"]}\n\n')

        # 尝试下载 NOTICE 文件内容
        notice_url = f'https://raw.githubusercontent.com/{dependency["name"]}/{dependency["version"]}/NOTICE'
        try:
            response = requests.get(notice_url)
            if response.ok:
                f.write(response.text.strip() + '\n\n')
        except:
            pass

        f.write('\n')