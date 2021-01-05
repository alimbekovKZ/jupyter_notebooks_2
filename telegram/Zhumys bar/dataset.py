api_id = 932597
api_hash = 'c2ba7241f9d3229c5fbd57a89dfe245b'
phone = '+77014388289'

import telethon
import pandas as pd
print(telethon.__version__)


from telethon import TelegramClient


from telethon import TelegramClient, events, sync


client = TelegramClient('session_name', api_id, api_hash)
client.start()
channel_username = 'https://t.me/zhumysbar' # your channel
ids = []
message = []
mes_date = []
view = []
mes_title = []
mes_description = []
for mes in client.get_messages(channel_username, limit=None):
    try:
        ids.append(mes.id)
        message.append(mes.message)
        mes_date.append(mes.date)
        view.append(mes.views)
        mes_title.append(mes.media.webpage.title)
        mes_description.append(mes.media.webpage.description)
    except:
        #message.append('')
        #mes_date.append('')
        #view.append(0)
        mes_title.append('None')
        mes_description.append('None')


print(len(ids), len(message), len(mes_date), len(view), len(mes_title), len(mes_description))

corpus = pd.DataFrame(data = {
    'id': ids,
    'text': message, #[mes.message for mes in work_messages],
    'mes_date': mes_date, #[mes.date for mes in work_messages],
    'views': view, #[mes.views for mes in work_messages],
    'mes_title': mes_title, #[mes.media.webpage.title for mes in work_messages],
    'mes_description': mes_description, #[mes.media.webpage.description for mes in work_messages],
})

print(corpus.shape)

print(corpus.head(2))

corpus.to_csv('zhumysbar.csv', index = False)
