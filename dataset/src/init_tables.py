from sqlalchemy import text


def init_database(engine):
    """
    Инициализация всех таблиц в базе данных
    """
    try:
        with engine.begin() as conn:
            # Создание таблицы channels если не существует
            conn.execute(text("""
                CREATE TABLE public.channels (
                id serial4 NOT NULL,
                "name" varchar(255) NOT NULL,
                subscribers int4 NULL,
                url text NOT NULL,
                types_reactions _text NULL,
                media bool NULL,
                lang varchar(2) NULL,
                date_begin date NULL,
                used bool NOT NULL,
                processing_depth int4 NOT NULL,
                "comment" text NULL,
                CONSTRAINT channels_pkey PRIMARY KEY (id)
                );
            """))

            # Создание таблицы narratives если не существует
            conn.execute(text("""
                CREATE TABLE public.narratives (
                    id uuid DEFAULT gen_random_uuid() NOT NULL,
                    message_id int4 NOT NULL,
                    "date" timestamp NOT NULL,
                    message text NOT NULL,
                    id_channel int4 NOT NULL,
                    message_vector tsvector GENERATED ALWAYS AS (to_tsvector('russian'::regconfig, message)) STORED NULL,
                    CONSTRAINT narratives_pkey PRIMARY KEY (id)
                );
                CREATE INDEX idx_narratives_message_search ON public.narratives USING gin (message_vector);
                CREATE UNIQUE INDEX narratives_message_id_idx ON public.narratives (message_id,id_channel);
                ALTER TABLE public.narratives ADD CONSTRAINT narratives_id_channel_fkey FOREIGN KEY (id_channel) REFERENCES public.channels(id);
                );
            """))

            # Создание основной партиционированной таблицы message_day
            conn.execute(text("""
                CREATE TABLE public.message_day (
                    id uuid NOT NULL,
                    id_channel int4 NOT NULL,
                    cur_date date NOT NULL,
                    message_id uuid NOT NULL,
                    "views" int4 NOT NULL,
                    forwards int4 NOT NULL,
                    replies int4 NOT NULL,
                    reactions int4 NOT NULL,
                    delta_views int4 NOT NULL,
                    delta_forwards int4 NOT NULL,
                    delta_replies int4 NOT NULL,
                    delta_reactions int4 NOT NULL,
                    CONSTRAINT message_day_pkey PRIMARY KEY (id, cur_date),
                    CONSTRAINT fk_message_day_channel FOREIGN KEY (id_channel) REFERENCES public.channels(id),
                    CONSTRAINT fk_message_narratives FOREIGN KEY (message_id) REFERENCES public.narratives(id)
                )
                PARTITION BY RANGE (cur_date);

                CREATE TABLE public.message_day_2024_01 PARTITION OF public.message_day  FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

                CREATE TABLE public.message_day_2024_02 PARTITION OF public.message_day  FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

                CREATE TABLE public.message_day_2024_03 PARTITION OF public.message_day  FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

                CREATE TABLE public.message_day_2024_04 PARTITION OF public.message_day  FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');

                CREATE TABLE public.message_day_2024_05 PARTITION OF public.message_day  FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');

                CREATE TABLE public.message_day_2024_06 PARTITION OF public.message_day  FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');

                CREATE TABLE public.message_day_2024_07 PARTITION OF public.message_day  FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');

                CREATE TABLE public.message_day_2024_08 PARTITION OF public.message_day  FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');

                CREATE TABLE public.message_day_2024_09 PARTITION OF public.message_day  FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');

                CREATE TABLE public.message_day_2024_10 PARTITION OF public.message_day  FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');

                CREATE TABLE public.message_day_2024_11 PARTITION OF public.message_day  FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');

                CREATE TABLE public.message_day_2024_12 PARTITION OF public.message_day  FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

                CREATE TABLE public.message_day_2025_01 PARTITION OF public.message_day  FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

                CREATE TABLE public.message_day_2025_02 PARTITION OF public.message_day  FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

                CREATE TABLE public.message_day_2025_03 PARTITION OF public.message_day  FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

                CREATE TABLE public.message_day_2025_04 PARTITION OF public.message_day  FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');

                CREATE TABLE public.message_day_2025_05 PARTITION OF public.message_day  FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');

                CREATE TABLE public.message_day_2025_06 PARTITION OF public.message_day  FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');

                CREATE TABLE public.message_day_2025_07 PARTITION OF public.message_day  FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

                CREATE TABLE public.message_day_2025_08 PARTITION OF public.message_day  FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');

                CREATE TABLE public.message_day_2025_09 PARTITION OF public.message_day  FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

                CREATE TABLE public.message_day_2025_10 PARTITION OF public.message_day  FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

                CREATE TABLE public.message_day_2025_11 PARTITION OF public.message_day  FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

                CREATE TABLE public.message_day_2025_12 PARTITION OF public.message_day  FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');
            """))

    except Exception as e:
        print(f"Ошибка при инициализации базы данных: {str(e)}")
        raise