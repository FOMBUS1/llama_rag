from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from os import getenv

import asyncio
import env
import json
import api


TOKEN = getenv("BOT_TOKEN")

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Привет! Я виртуальный помощник! Меня зовут HistoryAI. Чем я могу тебе помочь?")

@dp.message()
async def echo_handler(message: Message) -> None:
    answer = await api.get_asnwer(message.text)

    await message.answer(answer)


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())