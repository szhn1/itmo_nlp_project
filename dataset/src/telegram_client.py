import asyncio
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from sqlalchemy import and_, text, func
from datetime import datetime
from typing import List, Dict
from src.models import Narratives, MessagesDay
from src.exceptions import (
    TelegramConnectionError, TelegramAuthError,
    TelegramChannelError, TelegramMessageError
)

class TelegramParser:
    def __init__(self, api_id, api_hash, phone, db_manager, session_name="session_name", batch_size=100):
        self.client = TelegramClient(session_name, api_id, api_hash)
        self.batch_size = batch_size
        self.phone = phone
        self.db_manager = db_manager
        self.MAX_MESSAGES = 5000

    async def start(self):
        try:
            await self.client.start(phone=self.phone)
            print("Клиент подключен!")
        except Exception as e:
            raise TelegramAuthError(f"Ошибка аутентификации в Telegram: {e}")

    async def stop(self):
        """Остановка и отключение клиента Telegram"""
        try:
            await self.client.disconnect()
            print("Клиент отключен!")
        except Exception as e:
            raise TelegramConnectionError(f"Ошибка отключения от Telegram: {e}")

    def check_channel_processed(self, session, channel_id, current_date):
        """Проверка, был ли уже обработан канал сегодня"""
        exists = session.query(func.count(MessagesDay.id))\
            .filter(
                and_(
                    MessagesDay.id_channel == channel_id,
                    MessagesDay.cur_date == current_date
                )
            ).scalar() > 0
        return exists

    def get_previous_day_stats(self, session, message_id, current_date):
        """
        Получение статистики за предыдущий запуск системы.
        Ищет самую последнюю запись статистики до текущей даты.
        """
        previous_stats = session.query(MessagesDay)\
            .filter(
                and_(
                    MessagesDay.message_id == message_id,
                    MessagesDay.cur_date < current_date
                )
            )\
            .order_by(MessagesDay.cur_date.desc())\
            .first()
        return previous_stats

    def calculate_deltas(self, current_values, previous_stats):
        """Расчет дельт значений между текущим и предыдущим запуском"""
        if not previous_stats:
            return {
                'delta_views': 0,
                'delta_forwards': 0,
                'delta_replies': 0,
                'delta_reactions': 0
            }
        return {
            'delta_views': current_values['views'] - (previous_stats.views or 0),
            'delta_forwards': current_values['forwards'] - (previous_stats.forwards or 0),
            'delta_replies': current_values['replies'] - (previous_stats.replies or 0),
            'delta_reactions': current_values['reactions'] - (previous_stats.reactions or 0)
        }

    async def process_message(self, message, channel_id, session, narratives_batch: List[Narratives], messages_day_batch: List[MessagesDay]) -> bool:
        """Обработка отдельного сообщения и накопление данных для пакетного сохранения"""
        try:
            # Проверяем, что сообщение не пустое
            if not message.message or not message.message.strip():
                return False

            # Получаем существующую запись из Narratives
            existing_message = session.query(Narratives).filter(
                and_(
                    Narratives.message_id == message.id,
                    Narratives.id_channel == channel_id
                )
            ).first()

            # Если сообщение содержит nan - пропускаем
            if not message.message :
                return False

            current_system_date = datetime.now().date()

            if not self.db_manager.ensure_partition_exists(current_system_date):
                raise TelegramMessageError(f"Не удалось создать партицию для даты {current_system_date}")

            # Если сообщение новое и не пустое - создаем запись
            if not existing_message:
                narrative = Narratives(
                    message_id=message.id,
                    date=message.date,
                    message=message.message.strip(),
                    id_channel=channel_id
                )
                narratives_batch.append(narrative)
                session.add(narrative)
                session.flush()
            else:
                narrative = existing_message

            current_values = {
                'views': message.views if message.views is not None else 0,
                'forwards': message.forwards if message.forwards is not None else 0,
                'replies': message.replies.replies if message.replies is not None else 0,
                'reactions': len(message.reactions.results) if message.reactions else 0
            }

            previous_stats = self.get_previous_day_stats(session, narrative.id, current_system_date)
            deltas = self.calculate_deltas(current_values, previous_stats)

            message_day = MessagesDay(
                id=narrative.id,
                cur_date=current_system_date,
                message_id=narrative.id,
                id_channel=channel_id,
                views=current_values['views'],
                forwards=current_values['forwards'],
                replies=current_values['replies'],
                reactions=current_values['reactions'],
                delta_views=deltas['delta_views'],
                delta_forwards=deltas['delta_forwards'],
                delta_replies=deltas['delta_replies'],
                delta_reactions=deltas['delta_reactions']
            )
            messages_day_batch.append(message_day)
            return True

        except Exception as e:
            print(f"Ошибка обработки сообщения {message.id}: {e}")
            raise TelegramMessageError(f"Ошибка при обработке сообщения {message.id}: {e}")

    async def process_channel(self, channel, session):
        """Обработка отдельного канала с пакетным сохранением данных"""
        stats = {
            'total_processed': 0,  # Всего обработано сообщений
            'saved_messages': 0,   # Сохранено непустых сообщений
            'narratives_added': 0, # Новые записи в narratives
            'metrics_added': 0     # Записи метрик
        }

        try:
            current_system_date = datetime.now().date()

            # Проверяем, не был ли уже обработан канал сегодня
            if self.check_channel_processed(session, channel.id, current_system_date):
                print(f"Канал {channel.name} уже был обработан сегодня ({current_system_date}). Пропускаем.")
                return stats

            telegram_channel = await self.client.get_entity(channel.url)

            messages_to_process = min(channel.processing_depth, self.MAX_MESSAGES)

            offset_id = 0
            narratives_batch: List[Narratives] = []
            messages_day_batch: List[MessagesDay] = []

            while stats['total_processed'] < messages_to_process:
                print(f"Запрос сообщений для канала {channel.name}, начиная с offset_id={offset_id}...")

                try:
                    history = await self.client(GetHistoryRequest(
                        peer=telegram_channel,
                        limit=self.batch_size,
                        offset_date=None,
                        offset_id=offset_id,
                        max_id=0,
                        min_id=0,
                        add_offset=0,
                        hash=0
                    ))

                    if not history.messages:
                        break

                    for message in history.messages:
                        if stats['total_processed'] >= messages_to_process:
                            break

                        stats['total_processed'] += 1

                        success = await self.process_message(
                            message,
                            channel.id,
                            session,
                            narratives_batch,
                            messages_day_batch
                        )
                        if success:
                            stats['saved_messages'] += 1

                    if not history.messages:
                        break

                    offset_id = history.messages[-1].id
                    await asyncio.sleep(1)

                except Exception as e:
                    print(f"Ошибка получения истории сообщений: {e}")
                    await asyncio.sleep(5)
                    raise TelegramChannelError(f"Ошибка получения истории сообщений для канала {channel.name}: {e}")

            # Пакетное сохранение данных
            try:
                if narratives_batch:
                    session.bulk_save_objects(narratives_batch)
                    stats['narratives_added'] = len(narratives_batch)

                if messages_day_batch:
                    session.bulk_save_objects(messages_day_batch)
                    stats['metrics_added'] = len(messages_day_batch)

                session.commit()

            except Exception as e:
                session.rollback()
                raise TelegramChannelError(f"Ошибка сохранения данных канала {channel.name}: {e}")

            return stats

        except Exception as e:
            session.rollback()
            raise TelegramChannelError(f"Ошибка обработки канала {channel.name}: {e}")