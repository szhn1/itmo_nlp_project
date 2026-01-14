import asyncio
import schedule
import time
import os
import requests
from dotenv import load_dotenv
from src.database import DatabaseManager
from src.telegram_client import TelegramParser
from src.models import Channels
from src.exceptions import (
    DatabaseError, TelegramError, DatabaseConnectionError, TelegramChannelError,
    TelegramConnectionError, TelegramAuthError
)
from contextlib import contextmanager
from datetime import datetime

class TelegramParserApp: 
    def __init__(self):
        load_dotenv()

        # Конфигурация базы данных
        self.db_uri = self._get_db_uri()
        self.db_manager = DatabaseManager(self.db_uri)

        self.api_id = os.getenv("API_ID")
        self.api_hash = os.getenv("API_HASH")
        self.phone = os.getenv("PHONE")
        self.tg_token = os.getenv("TG_TOKEN")
        self.chat_ids = os.getenv("CHAT_IDS")

        # Создаем парсер только один раз при инициализации
        self.parser = TelegramParser(
            api_id=self.api_id,
            api_hash=self.api_hash,
            phone=self.phone,
            db_manager=self.db_manager,
        )

        # Добавляем статистику для хранения результатов обработки
        self.processing_stats = {
            'channels': {},
            'total': {
                'processed_messages': 0,
                'saved_messages': 0,
                'channels_processed': 0,
                'channels_failed': 0
            }
        }

        # Создаем и сохраняем loop при инициализации
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def send_monitoring_message(self, message):
        """Отправка сообщения в канал мониторинга"""
        try:
            async with asyncio.timeout(1):
                requests.get(
                    f'https://api.telegram.org/bot{self.tg_token}/sendMessage',
                    params=dict(chat_id=self.chat_ids, text=message),
                )
        except Exception as e:
            print(f"Ошибка отправки сообщения мониторинга: {e}")

    def _get_db_uri(self):
        required_env_vars = ['POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB', 'POSTGRES_HOST', 'POSTGRES_PORT']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise DatabaseConnectionError(f"Отсутствуют обязательные переменные окружения: {', '.join(missing_vars)}")

        return f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@" \
               f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

    @contextmanager
    def get_session(self):
        """Контекстный менеджер для сессий базы данных"""
        session = self.db_manager.get_session()
        try:
            yield session
        finally:
            session.close()

    async def process_channels(self):
        """Обработка всех активных каналов"""
        start_time = datetime.now()
        start_message = f"🚀 Начало обработки каналов\nВремя запуска: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        await self.send_monitoring_message(start_message)

        try:
            print("Начало обработки каналов...")
            self.db_manager.initialize_database()

            # Используем существующий парсер
            if not self.parser.client.is_connected():
                await self.parser.start()

            with self.get_session() as session:
                active_channels = session.query(Channels).filter(Channels.used == True).all()

                if not active_channels:
                    await self.send_monitoring_message("⚠️ Нет активных каналов для обработки")
                    return

                for channel in active_channels:
                    try:
                        channel_stats = {
                            'name': channel.name,
                            'url': channel.url,
                            'processed_messages': 0,
                            'saved_messages': 0,
                            'start_time': datetime.now(),
                            'status': 'success'
                        }

                        stats = await self.parser.process_channel(channel, session)

                        channel_stats.update({
                            'processed_messages': stats['total_processed'],
                            'saved_messages': stats['saved_messages'],
                            'end_time': datetime.now()
                        })

                        duration = channel_stats['end_time'] - channel_stats['start_time']
                        channel_message = (
                            f"📊 Статистика канала {channel.name}\n"
                            f"🔗 URL: {channel.url}\n"
                            f"📥 Обработано сообщений: {stats['total_processed']}\n"
                            f"💾 Сохранено сообщений: {stats['saved_messages']}\n"
                            f"⏱ Продолжительность: {str(duration).split('.')[0]}"
                        )
                        await self.send_monitoring_message(channel_message)

                        self.processing_stats['total']['processed_messages'] += stats['total_processed']
                        self.processing_stats['total']['saved_messages'] += stats['saved_messages']
                        self.processing_stats['total']['channels_processed'] += 1

                    except TelegramChannelError as e:
                        channel_stats['status'] = 'error'
                        channel_stats['error'] = str(e)
                        self.processing_stats['total']['channels_failed'] += 1
                        error_message = f"❌ Ошибка обработки канала {channel.name}: {e}"
                        await self.send_monitoring_message(error_message)
                        continue

                    finally:
                        self.processing_stats['channels'][channel.name] = channel_stats

        except (DatabaseError, TelegramError) as e:
            error_message = f"⚠️ Критическая ошибка: {e}"
            await self.send_monitoring_message(error_message)
            raise
        finally:
            # Не отключаем клиент после каждого запуска
            end_time = datetime.now()
            duration = end_time - start_time

            total_stats = self.processing_stats['total']
            completion_message = (
                f"📈 ИТОГОВАЯ СТАТИСТИКА\n\n"
                f"📊 Общие показатели:\n"
                f"- Обработано каналов: {total_stats['channels_processed']}\n"
                f"- Каналов с ошибками: {total_stats['channels_failed']}\n"
                f"- Всего обработано сообщений: {total_stats['processed_messages']}\n"
                f"- Всего сохранено сообщений: {total_stats['saved_messages']}\n\n"
                f"⏱ Информация о выполнении:\n"
                f"- Время начала: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"- Время окончания: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"- Общая продолжительность: {str(duration).split('.')[0]}"
            )
            await self.send_monitoring_message(completion_message)
            self.processing_stats = {
            'channels': {},
            'total': {
                'processed_messages': 0,
                'saved_messages': 0,
                'channels_processed': 0,
                'channels_failed': 0
            }
        }

    def schedule_job(self):
        """Планирование задачи"""
        try:
            self.loop.run_until_complete(self.process_channels())
        except (DatabaseError, TelegramError) as e:
            print(f"Ошибка выполнения задачи: {e}")
            print("Повторная попытка через час...")
            schedule.every(1).hours.do(self.schedule_job)

    async def init_monitoring(self):
        """Инициализация клиента мониторинга"""
        try:
            await self.send_monitoring_message("🔄 Система мониторинга инициализирована")
        except Exception as e:
            print(f"Ошибка инициализации мониторинга: {e}")

def main():
    app = TelegramParserApp()

    # Инициализация мониторинга
    app.loop.run_until_complete(app.init_monitoring())

    # Планируем выполнение задачи каждый день в указанное время
    schedule.every().day.at(os.getenv("TIME_START")).do(app.schedule_job)

    print("Запуск планировщика...")

    # Поддерживаем работу скрипта
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()