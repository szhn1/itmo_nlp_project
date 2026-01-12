from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import datetime
from dateutil.relativedelta import relativedelta
from src.models import Base
from src.exceptions import (
    DatabaseError, DatabaseConnectionError, DatabasePartitionError
)
import os

class DatabaseManager:
    def __init__(self, db_uri):
        try:
            self.engine = create_engine(db_uri, pool_pre_ping=True, connect_args={"options": "-c timezone=utc"})
            self.Session = sessionmaker(bind=self.engine)
        except Exception as e:
            raise DatabaseConnectionError(f"Ошибка подключения к базе данных: {e}")

    def is_channels_empty(self):
        """Проверяет, пуста ли таблица channels"""
        try:
            with self.engine.begin() as connection:
                result = connection.execute(text("SELECT COUNT(*) FROM channels")).scalar()
                return result == 0
        except Exception as e:
            raise DatabaseError(f"Ошибка проверки таблицы channels: {e}")

    def create_partition(self, connection, start_date):
        """Создает партицию для указанного месяца"""
        try:
            # Вычисляем дату окончания как первый день следующего месяца
            end_date = start_date + relativedelta(months=1)
            partition_name = f"message_day_{start_date.strftime('%Y_%m')}"

            sql = text(f"""
            CREATE TABLE IF NOT EXISTS {partition_name}
            PARTITION OF message_day
            FOR VALUES FROM (:start_date) TO (:end_date)
            """)

            connection.execute(sql, {
                'start_date': start_date,
                'end_date': end_date
            })

            print(f"Создана партиция {partition_name}")
        except Exception as e:
            raise DatabasePartitionError(f"Ошибка создания партиции для {start_date}: {e}")

    def ensure_partition_exists(self, date):
        """Проверяет существование партиции для указанной даты"""
        try:
            with self.engine.begin() as connection:
                start_date = date.replace(day=1)
                partition_name = f"message_day_{start_date.strftime('%Y_%m')}"

                check_partition_sql = text("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_catalog.pg_class c
                        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = 'public'
                        AND c.relname = :partition_name
                    );
                """)

                result = connection.execute(check_partition_sql, {'partition_name': partition_name}).scalar()

                if not result:
                    self.create_partition(connection, start_date)
                    print(f"Создана новая партиция для {start_date}")
                return True
        except Exception as e:
            raise DatabasePartitionError(f"Ошибка проверки существования партиции для {date}: {e}")

    def create_future_partitions(self, months_ahead=12):
        """Создает партиции на несколько месяцев вперед"""
        try:
            with self.engine.begin() as connection:
                moscow_tz = datetime.timezone(datetime.timedelta(hours=3))
                current_date = datetime.datetime.now(moscow_tz).date()
                start_date = current_date.replace(day=1)

                for _ in range(months_ahead):
                    self.ensure_partition_exists(start_date)
                    start_date = start_date + relativedelta(months=1)

        except Exception as e:
            raise DatabasePartitionError(f"Ошибка создания будущих партиций: {e}")

    def execute_sql_script(self, script_path):
        """Выполняет SQL скрипт из файла"""
        try:
            if not os.path.exists(script_path):
                raise DatabaseError(f"SQL скрипт не найден: {script_path}")

            with open(script_path, 'r', encoding='utf-8') as file:
                sql_script = file.read()

            with self.engine.begin() as connection:
                connection.execute(text(sql_script))
                print(f"SQL скрипт {script_path} успешно выполнен")
        except Exception as e:
            raise DatabaseError(f"Ошибка выполнения SQL скрипта: {e}")

    def initialize_database(self):
        """Проверяет существование таблиц и создает их при необходимости"""
        try:
            inspector = inspect(self.engine)
            Base.metadata.create_all(self.engine)
            self.create_future_partitions()

            # Выполняем скрипт добавления каналов только если таблица пуста
            if self.is_channels_empty():
                script_path = os.path.join('src', 'add_chanels.sql')
                self.execute_sql_script(script_path)
                print("Каналы успешно добавлены из SQL скрипта")

            print("Таблицы и партиции успешно созданы!")
        except Exception as e:
            raise DatabaseError(f"Ошибка проверки/создания таблиц: {e}")

    def get_session(self):
        """Создает и возвращает новую сессию"""
        try:
            return self.Session()
        except Exception as e:
            raise DatabaseConnectionError(f"Ошибка создания сессии: {e}")