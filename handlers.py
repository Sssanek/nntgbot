from main import bot, dp
from get_result import start_style
from aiogram.types import Message
from config import admin_id


async def send_to_admin(dp):
    await bot.send_message(chat_id=admin_id, text='Бот запущен')


"""@dp.message_handler()
async def echo(message: Message):
    text = f"Привет, ты написал: {message.text}"
    await bot.send_message(chat_id=message.from_user.id, text=text)"""


images = dict()
flag = 0



@dp.message_handler(content_types=['text'])
async def start(message):
    if message.text == '/go':
        await bot.send_message(message.from_user.id,
                         "Чтобы перенести стиль, отправьте два фото.\n" +
                         "1-я картинка - это исходное изображение, 2-я картинка стиль который будет перенесен")
        # bot.register_next_step_handler(message, handle_docs_photo)
    else:
        await bot.send_message(message.from_user.id, 'Напиши /go')


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message):
    global flag
    if flag == 0:
        await message.photo[-1].download('content.jpg')
        await bot.send_message(message.from_user.id, 'Исходное изображение загружено')
        flag = 1
        # примем первую пикчу
    elif flag == 1:
        await message.photo[-1].download('style.jpg')
        await bot.send_message(message.from_user.id, 'Изображение стиля загружено')
        await bot.send_message(message.from_user.id, 'Ожидайте результата')
        flag = 2
        # примем вторую пикчу
    if flag == 2:
        reply_img = start_style('style.jpg', 'content.jpg')
        await bot.send_photo(message.chat.id, open(reply_img, 'rb'))
        await bot.send_message(message.from_user.id, 'А вот и результат')
        flag = 0
        # делаем прикол с изображениями


"""def progress(cur, total):
    percent = round(cur / total) * 100
    bot.send_message(message.from_user.id, f'Обработка изображения {percent}%')"""

