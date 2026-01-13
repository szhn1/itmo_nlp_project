class DatabaseError(Exception):
    """Базовый класс для исключений базы данных"""
    pass

class DatabaseConnectionError(DatabaseError):
    """Ошибка подключения к базе данных"""
    pass

class DatabaseQueryError(DatabaseError):
    """Ошибка выполнения запроса к базе данных"""
    pass

class DatabasePartitionError(DatabaseError):
    """Ошибка создания или работы с партициями"""
    pass

class TelegramError(Exception):
    """Базовый класс для исключений Telegram"""
    pass

class TelegramConnectionError(TelegramError):
    """Ошибка подключения к Telegram API"""
    pass

class TelegramAuthError(TelegramError):
    """Ошибка аутентификации в Telegram"""
    pass

class TelegramChannelError(TelegramError):
    """Ошибка при работе с каналом Telegram"""
    pass

class TelegramMessageError(TelegramError):
    """Ошибка при обработке сообщения Telegram"""
    pass