import logging
import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler,
    PicklePersistence,
)
from telegram.error import BadRequest, NetworkError
import requests
import json
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, Optional, List, Tuple
import html
import traceback
import sqlite3
from sqlite3 import Error
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import numpy as np
from matplotlib import patheffects
from matplotlib import rcParams
from countries_data import COUNTRIES_INFO

# Настройка шрифтов для matplotlib
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Helvetica', 'Verdana']

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://ollama:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
ADMIN_IDS = [int(id) for id in os.getenv("ADMIN_IDS", "").split(",") if id]
DATA_FILE = os.getenv("DATA_FILE", "bot_data.pkl")
DB_FILE = os.getenv("DB_FILE", "bot_stats.db")

if not TELEGRAM_TOKEN:
    raise ValueError("Токен бота не найден! Создайте файл .env с TELEGRAM_TOKEN")

# Состояния для ConversationHandler
SELECTING_COUNTRY, SELECTING_FEATURE = range(2)

class DatabaseManager:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self._create_tables()
    
    def _create_connection(self):
        """Создать соединение с базой данных SQLite"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            return conn
        except Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _create_tables(self):
        """Создать необходимые таблицы в базе данных"""
        sql_create_country_stats_table = """
        CREATE TABLE IF NOT EXISTS country_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            country TEXT NOT NULL,
            feature TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        sql_create_feedback_table = """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            feedback TEXT NOT NULL,
            is_positive INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            conn = self._create_connection()
            cursor = conn.cursor()
            cursor.execute(sql_create_country_stats_table)
            cursor.execute(sql_create_feedback_table)
            conn.commit()
        except Error as e:
            logger.error(f"Error creating tables: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def log_country_selection(self, user_id: int, country: str, feature: Optional[str] = None):
        """Записать выбор страны пользователем"""
        sql = """
        INSERT INTO country_stats(user_id, country, feature)
        VALUES(?, ?, ?)
        """
        
        try:
            conn = self._create_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (user_id, country, feature))
            conn.commit()
            logger.info(f"Logged country selection for user {user_id}: {country} - {feature}")
        except Error as e:
            logger.error(f"Error logging country selection: {e}")
        finally:
            if conn:
                conn.close()
    
    def log_feedback(self, user_id: int, feedback: str, is_positive: bool):
        """Записать отзыв пользователя"""
        sql = """
        INSERT INTO feedback(user_id, feedback, is_positive)
        VALUES(?, ?, ?)
        """
        
        try:
            conn = self._create_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (user_id, feedback, 1 if is_positive else 0))
            conn.commit()
            logger.info(f"Logged feedback from user {user_id}")
        except Error as e:
            logger.error(f"Error logging feedback: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_country_stats(self, period: str = "all") -> Dict[str, int]:
        """Получить статистику по странам за указанный период"""
        period_conditions = {
            "day": "timestamp >= datetime('now', '-1 day')",
            "week": "timestamp >= datetime('now', '-7 days')",
            "month": "timestamp >= datetime('now', '-1 month')",
            "all": "1=1"
        }
        
        condition = period_conditions.get(period, "1=1")
        
        sql = f"""
        SELECT country, COUNT(*) as count
        FROM country_stats
        WHERE {condition}
        GROUP BY country
        ORDER BY count DESC
        """
        
        stats = {}
        try:
            conn = self._create_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            for row in rows:
                stats[row["country"]] = row["count"]
        except Error as e:
            logger.error(f"Error getting country stats: {e}")
        finally:
            if conn:
                conn.close()
        
        return stats
    
    def get_feature_stats(self, country: str, period: str = "all") -> Dict[str, int]:
        """Получить статистику по функциям для конкретной страны"""
        period_conditions = {
            "day": "timestamp >= datetime('now', '-1 day')",
            "week": "timestamp >= datetime('now', '-7 days')",
            "month": "timestamp >= datetime('now', '-1 month')",
            "all": "1=1"
        }
        
        condition = period_conditions.get(period, "1=1")
        
        sql = f"""
        SELECT feature, COUNT(*) as count
        FROM country_stats
        WHERE country = ? AND feature IS NOT NULL AND {condition}
        GROUP BY feature
        ORDER BY count DESC
        """
        
        stats = {}
        try:
            conn = self._create_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (country,))
            rows = cursor.fetchall()
            
            for row in rows:
                stats[row["feature"]] = row["count"]
        except Error as e:
            logger.error(f"Error getting feature stats: {e}")
        finally:
            if conn:
                conn.close()
        
        return stats

# Инициализация менеджера базы данных
db_manager = DatabaseManager(DB_FILE)

class UserDataManager:
    def __init__(self):
        self.user_data = {}
    
    def set_country(self, user_id: int, country: str) -> None:
        if user_id not in self.user_data:
            self.user_data[user_id] = {}
        self.user_data[user_id]["selected_country"] = country
        db_manager.log_country_selection(user_id, country)
    
    def get_country(self, user_id: int) -> Optional[str]:
        return self.user_data.get(user_id, {}).get("selected_country")
    
    def add_feedback(self, user_id: int, feedback: str, is_positive: bool) -> None:
        if user_id not in self.user_data:
            self.user_data[user_id] = {}
        if "feedback" not in self.user_data[user_id]:
            self.user_data[user_id]["feedback"] = []
        self.user_data[user_id]["feedback"].append(feedback)
        db_manager.log_feedback(user_id, feedback, is_positive)

user_data_manager = UserDataManager()

async def generate_with_ollama(prompt: str, country: Optional[str] = None) -> str:
    """Генерация ответа с помощью Ollama API"""
    try:
        full_prompt = (
            "Ответь на русском языке кратко и информативно (максимум 2 абзаца). "
            "Будь дружелюбным и полезным. "
        )
        
        if country:
            full_prompt += f"Учитывай, что вопрос относится к стране {country}. "
        
        full_prompt += f"Вопрос: {prompt}"
        
        data = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_ctx": 2048,
                "top_p": 0.9
            }
        }
        
        response = requests.post(
            OLLAMA_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "Не удалось получить ответ. Пожалуйста, попробуйте позже.")
        
        return html.escape(response_text)
    
    except requests.exceptions.Timeout:
        logger.warning("Ollama API timeout")
        return "Извините, сервис временно недоступен из-за таймаута. Пожалуйста, попробуйте позже."
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API error: {e}")
        return "Извините, возникла проблема с сервисом. Пожалуйста, попробуйте позже."
    except Exception as e:
        logger.error(f"Unexpected error in generate_with_ollama: {e}")
        return "Произошла непредвиденная ошибка при обработке вашего запроса."

async def create_stats_plot(stats: Dict[str, int], title: str) -> BytesIO:
    """Создать красивый график статистики с современным дизайном"""
    if not stats:
        return None

    # Настройка matplotlib для работы без GUI
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Установка стиля
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'
    
    # Подготовка данных
    sorted_stats = dict(sorted(stats.items(), key=lambda item: item[1], reverse=True))
    categories = list(sorted_stats.keys())
    values = list(sorted_stats.values())
    
    # Создание фигуры с красивым фоном
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Цветовая палитра
    colors = sns.color_palette("viridis", len(categories))
    
    # Горизонтальный бар-чарт с тенью
    bars = ax.barh(
        y=categories,
        width=values,
        color=colors,
        height=0.7,
        edgecolor='white',
        linewidth=1.5
    )
    
    # Добавление тени для эффекта глубины
    for bar in bars:
        bar.set_path_effects([
            patheffects.withSimplePatchShadow(
                offset=(3, -3), 
                shadow_rgbFace='#333F4B', 
                alpha=0.2
            )
        ])
    
    # Удаление верхней и правой границ
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Настройка осей
    ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='#d3d3d3')
    ax.yaxis.grid(False)
    
    # Настройка подписей
    ax.set_xlabel('Количество запросов', fontsize=12, color='#333F4B')
    ax.set_ylabel('', fontsize=12, color='#333F4B')
    
    # Красивый заголовок
    plt.title(
        title,
        fontsize=16,
        pad=20,
        fontweight='bold',
        color='#333F4B'
    )
    
    # Добавление значений на график
    for i, (category, value) in enumerate(zip(categories, values)):
        ax.text(
            value + max(values)*0.01,  # Отступ от столбца
            i,                         # Позиция по y
            f'{value}',
            ha='left',
            va='center',
            fontsize=11,
            color='#333F4B',
            fontweight='bold'
        )
    
    # Добавление элегантной аннотации
    plt.figtext(
        0.95, 0.02,
        'Generated by TravelBot',
        ha='right',
        fontsize=10,
        color='#7f7f7f',
        alpha=0.7
    )
    
    # Оптимизация расположения элементов
    plt.tight_layout()
    
    # Сохранение в буфер
    buf = BytesIO()
    plt.savefig(
        buf,
        format='png',
        bbox_inches='tight',
        dpi=120,
        transparent=False
    )
    buf.seek(0)
    plt.close()
    
    return buf

async def show_country_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать статистику по странам с красивым графиком"""
    query = update.callback_query
    await query.answer()
    
    period = "all"
    if query.data.startswith("stats_"):
        period = query.data.split("_")[1]
    
    period_titles = {
        "day": "за последний день",
        "week": "за последнюю неделю",
        "month": "за последний месяц",
        "all": "за все время"
    }
    
    stats = db_manager.get_country_stats(period)
    
    if not stats:
        # Удаляем предыдущее сообщение, если оно было фото
        try:
            await query.message.delete()
        except BadRequest:
            pass
            
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"📊 Нет данных о выборе стран {period_titles.get(period, '')}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌍 Выбрать страну", callback_data="select_country")],
                [InlineKeyboardButton("🏠 На главную", callback_data="start")]
            ])
        )
        return
    
    # Создаем красивый график
    title = f"📈 Популярность стран {period_titles.get(period, '')}"
    plot_buf = await create_stats_plot(stats, title)
    
    if plot_buf:
        keyboard = [
            [
                InlineKeyboardButton("🕒 День", callback_data="stats_day"),
                InlineKeyboardButton("🕒 Неделя", callback_data="stats_week"),
                InlineKeyboardButton("🕒 Месяц", callback_data="stats_month")
            ],
            [InlineKeyboardButton("🌍 Выбрать страну", callback_data="select_country")],
            [InlineKeyboardButton("🏠 На главную", callback_data="start")]
        ]
        
        try:
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=plot_buf,
                caption=f"<b>{title}</b>\n\nНаведите на график для деталей",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="HTML"
            )
            await query.message.delete()
        except Exception as e:
            logger.error(f"Error sending stats plot: {e}")
            await query.edit_message_text(
                text="Ошибка при создании графика статистики",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 На главную", callback_data="start")]
                ])
            )

async def show_feature_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать статистику по функциям для выбранной страны"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    country = user_data_manager.get_country(user_id)
    
    if not country:
        # Удаляем предыдущее сообщение (график), если оно есть
        try:
            await query.message.delete()
        except BadRequest:
            pass
            
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Сначала выберите страну",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌍 Выбрать страну", callback_data="select_country")]
            ])  # Закрыли InlineKeyboardMarkup и список кнопок
        )
        return
    
    stats = db_manager.get_feature_stats(country)
    
    if not stats:
        # Удаляем предыдущее сообщение (график), если оно есть
        try:
            await query.message.delete()
        except BadRequest:
            pass

        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"📊 Нет данных по функциям для {country}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Назад", callback_data=f"back_to_{country}")],
                [InlineKeyboardButton("🏠 На главную", callback_data="start")]
            ])  # Закрыли InlineKeyboardMarkup
        )
        return
    
    # Создаем красивый график
    emoji = COUNTRIES_INFO[country].get("emoji", "🌐")
    title = f"📊 Популярность функций для {emoji} {country}"
    plot_buf = await create_stats_plot(stats, title)
    
    if plot_buf:
        keyboard = [
            [InlineKeyboardButton("🔙 Назад", callback_data=f"back_to_{country}")],
            [InlineKeyboardButton("🏠 На главную", callback_data="start")]
        ]
        
        try:
            # Сначала удаляем предыдущее сообщение (если это фото)
            try:
                await query.message.delete()
            except BadRequest:
                pass
                
            # Отправляем новый график
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=plot_buf,
                caption=f"<b>{title}</b>",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Error sending feature stats plot: {e}")
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="Ошибка при создании графика статистики",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 На главную", callback_data="start")]
                ])
            )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка команды /start"""
    return await send_new_start_message(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда помощи с современным дизайном"""
    help_text = (
        "📚 <b>Справка по использованию бота</b>\n\n"
        "<b>Основные команды:</b>\n"
        "/start - Начать диалог с ботом\n"
        "/help - Показать эту справку\n"
        "/ask - Задать произвольный вопрос\n"
        "/country - Выбрать страну\n"
        "/stats - Показать статистику\n\n"
        "<b>Примеры вопросов:</b>\n"
        "• Какие документы нужны для визы в Германию?\n"
        "• Как сказать 'спасибо' по-японски?\n"
        "• Какие чаевые приняты в США?\n"
        "• Какая сейчас дата в Токио?\n\n"
        "Просто напишите мне свой вопрос, и я постараюсь помочь!"
    )
    
    keyboard = [
        [InlineKeyboardButton("🌍 Выбрать страну", callback_data="select_country")],
        [InlineKeyboardButton("❓ Задать вопрос", callback_data="ask_question")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats_all")],
        [InlineKeyboardButton("🔙 На главную", callback_data="start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.message:
        await update.message.reply_text(help_text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.callback_query.edit_message_text(
            help_text, 
            reply_markup=reply_markup, 
            parse_mode="HTML"
        )

async def show_country_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Показать список стран с эмодзи и современным дизайном"""
    keyboard = []
    
    # Сортируем страны по алфавиту
    countries = sorted(COUNTRIES_INFO.keys(), key=lambda x: x.lower())
    
    # Группируем страны по 2 в ряд
    for i in range(0, len(countries), 2):
        row = []
        for country in countries[i:i+2]:
            emoji = COUNTRIES_INFO[country].get("emoji", "🌐")
            row.append(InlineKeyboardButton(
                f"{emoji} {country}", 
                callback_data=f"country_{country}"
            ))
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("📊 Статистика", callback_data="stats_all")])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="start")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = (
        "🌍 <b>Выберите страну:</b>\n\n"
        "Я могу предоставить подробную информацию о следующих странах:"
    )
    
    if hasattr(update, 'callback_query'):
        await update.callback_query.edit_message_text(
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML"
        )
    
    return SELECTING_COUNTRY

async def select_country(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора страны"""
    query = update.callback_query
    await query.answer()
    
    country = query.data.split("_")[1]
    user_id = update.effective_user.id
    
    # Сохраняем выбранную страну
    user_data_manager.set_country(user_id, country)
    
    logger.info(f"User {user_id} selected country: {country}")
    
    await show_country_features(update, country, context)
    return SELECTING_FEATURE

async def show_country_features(update: Update, country: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать функции для выбранной страны с надежной навигацией"""
    try:
        emoji = COUNTRIES_INFO[country].get("emoji", "🌐")
        
        keyboard = [
            [
                InlineKeyboardButton("🛂 Виза", callback_data=f"feature_visa"),
                InlineKeyboardButton("🗣️ Язык", callback_data=f"feature_language")
            ],
            [
                InlineKeyboardButton("🆘 Экстренное", callback_data=f"feature_emergency"),
                InlineKeyboardButton("🕒 Время", callback_data=f"feature_time")
            ],
            [
                InlineKeyboardButton("💰 Валюта", callback_data=f"feature_currency"),
                InlineKeyboardButton("🎭 Культура", callback_data=f"feature_culture")
            ],
            [InlineKeyboardButton("📊 Статистика", callback_data=f"feature_stats_{country}")],
            [InlineKeyboardButton("❓ Задать вопрос", callback_data="ask_question")],
            [
                InlineKeyboardButton("🔙 К списку стран", callback_data="select_country"),
                InlineKeyboardButton("🏠 На главную", callback_data="start")
            ]
        ]
        
        text = f"{emoji} <b>Вы выбрали:</b> {country}\n\nЧто вас интересует?"
        
        # Всегда отправляем новое сообщение вместо редактирования
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
        
        # Пытаемся удалить предыдущее сообщение (если есть)
        try:
            if hasattr(update, 'callback_query') and update.callback_query.message:
                await update.callback_query.message.delete()
        except BadRequest as e:
            logger.debug(f"Не удалось удалить сообщение: {e}")
            
    except Exception as e:
        logger.error(f"Ошибка в show_country_features: {e}")
        raise

async def show_feature_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать информацию о функции с современным дизайном"""
    query = update.callback_query
    await query.answer()
    
    data = query.data.split("_")
    feature = data[1]
    user_id = update.effective_user.id
    country = user_data_manager.get_country(user_id)
    
    if not country:
        await query.edit_message_text(
            "Сначала выберите страну",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌍 Выбрать страну", callback_data="select_country")]
            ])
        )
        return
    
    if feature == "time":
        await show_time(update, country, context)
        return
    
    if feature == "stats":
        await show_feature_stats(update, context)
        return
    
    country_info = COUNTRIES_INFO.get(country, {})
    
    if feature in country_info:
        info = country_info[feature]
    else:
        try:
            await query.edit_message_text("🔍 Ищу информацию...")
            feature_translations = {
                "visa": "визовом режиме",
                "language": "языке",
                "emergency": "экстренных службах",
                "currency": "валюте",
                "culture": "культурных особенностях"
            }
            feature_name = feature_translations.get(feature, feature)
            info = await generate_with_ollama(f"Расскажи о {feature_name} в {country}?", country)
        except Exception as e:
            logger.error(f"Error generating info: {e}")
            info = f"Не удалось получить информацию о {feature} для {country}"
    
    # Логируем выбор функции
    db_manager.log_country_selection(user_id, country, feature)
    
    # Форматирование текста
    formatted_info = info
    if feature == "emergency":
        formatted_info = info.replace("- ", "🆘 ").replace("\n", "\n\n")
    elif feature == "language":
        formatted_info = info.replace("- ", "🗣 ").replace("\n", "\n\n")
    
    # Источник информации
    source_note = "\n\nℹ️ Информация предоставлена официальными источниками" if feature in country_info else "\n\nℹ️ Информация сгенерирована ИИ"
    
    keyboard = [
        [InlineKeyboardButton("🔙 Назад", callback_data=f"back_to_{country}")],
        [InlineKeyboardButton("🏠 На главную", callback_data="start")]
    ]
    
    feature_titles = {
        "visa": "🛂 Визовый режим",
        "language": "🗣️ Язык и общение",
        "emergency": "🆘 Экстренные службы",
        "currency": "💰 Валюта и финансы",
        "culture": "🎭 Культурные особенности"
    }
    
    emoji = COUNTRIES_INFO[country].get("emoji", "🌐")
    
    try:
        await query.edit_message_text(
            text=(
                f"{emoji} <b>{country} - {feature_titles.get(feature, feature)}</b>\n\n"
                f"{formatted_info}\n"
                f"{source_note}"
            ),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML",
            disable_web_page_preview=True
        )
    except BadRequest as e:
        if "Message is not modified" in str(e):
            await query.answer("Информация уже актуальна")
        else:
            raise e

async def show_time(update: Update, country: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать текущее время с современным дизайном"""
    query = update.callback_query
    await query.answer()
    
    country_info = COUNTRIES_INFO.get(country, {})
    timezone = country_info.get("timezone", "UTC")
    emoji = country_info.get("emoji", "🌐")
    
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        
        weekdays = {
            "Monday": "Понедельник",
            "Tuesday": "Вторник",
            "Wednesday": "Среда",
            "Thursday": "Четверг",
            "Friday": "Пятница",
            "Saturday": "Суббота",
            "Sunday": "Воскресенье"
        }
        
        weekday_ru = weekdays.get(current_time.strftime('%A'), current_time.strftime('%A'))
        
        time_info = (
            f"⏰ <b>Текущее время в {emoji} {country}:</b>\n\n"
            f"• <b>Часы:</b> {current_time.strftime('%H:%M')}\n"
            f"• <b>Дата:</b> {current_time.strftime('%d.%m.%Y')}\n"
            f"• <b>День недели:</b> {weekday_ru}\n"
            f"• <b>Часовой пояс:</b> {timezone}\n\n"
            f"🔄 Обновлено: {datetime.now().strftime('%H:%M:%S')}"
        )
    except Exception as e:
        logger.error(f"Timezone error for {country}: {e}")
        time_info = f"Не удалось определить время для {country}"
    
    # Логируем запрос времени
    db_manager.log_country_selection(update.effective_user.id, country, "time")
    
    timestamp = int(datetime.now().timestamp())
    keyboard = [
        [InlineKeyboardButton("🔄 Обновить время", callback_data=f"feature_time_{timestamp}")],
        [InlineKeyboardButton("🔙 Назад", callback_data=f"back_to_{country}")]
    ]
    
    try:
        await query.edit_message_text(
            text=time_info,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
    except BadRequest as e:
        if "Message is not modified" in str(e):
            await query.answer("Время не изменилось с последней проверки")
        else:
            logger.error(f"Error editing time message: {e}")
            await query.answer("Ошибка при обновлении времени")

async def ask_question_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик вопросов с современным дизайном"""
    query = update.callback_query
    user_id = update.effective_user.id
    country = user_data_manager.get_country(user_id)
    
    prompt_text = "💬 <b>Напишите ваш вопрос:</b>\n\nЯ постараюсь помочь!"
    if country:
        prompt_text += f"\n\nСейчас выбрана страна: {COUNTRIES_INFO[country].get('emoji', '🌐')} {country}"
    
    keyboard = [
        [InlineKeyboardButton("❌ Отмена", callback_data="cancel_question")]
    ]
    
    if query:
        await query.answer()
        await query.edit_message_text(
            prompt_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(
            prompt_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
    
    return SELECTING_FEATURE

async def handle_user_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка вопроса пользователя с современным дизайном"""
    user_id = update.effective_user.id
    question = update.message.text.strip()
    country = user_data_manager.get_country(user_id)
    
    if not question:
        await update.message.reply_text("Пожалуйста, задайте ваш вопрос более подробно.")
        return SELECTING_FEATURE
    
    await update.message.reply_chat_action(action="typing")
    
    try:
        answer = await generate_with_ollama(question, country)
        
        if country:
            answer = (
                f"{COUNTRIES_INFO[country].get('emoji', '🌐')} <b>Контекст:</b> {country}\n\n"
                f"{answer}"
            )
        
        answer += "\n\n🔹 Этот ответ был полезен?"
        
        keyboard = [
            [
                InlineKeyboardButton("👍 Да", callback_data=f"feedback_yes_{hash(question)}"),
                InlineKeyboardButton("👎 Нет", callback_data=f"feedback_no_{hash(question)}")
            ],
            [InlineKeyboardButton("🌍 Выбрать другую страну", callback_data="select_country")],
            [InlineKeyboardButton("❓ Задать другой вопрос", callback_data="ask_question")],
            [InlineKeyboardButton("🏠 На главную", callback_data="start")]
        ]
        
        await update.message.reply_text(
            answer,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
        
    except Exception as e:
        logger.error(f"Error handling question: {e}")
        await update.message.reply_text(
            "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте позже.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 На главную", callback_data="start")]
            ]),
            parse_mode="HTML"
        )
    
    return SELECTING_FEATURE

async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка отзыва с современным дизайном"""
    query = update.callback_query
    await query.answer()
    
    data = query.data.split("_")
    feedback_type = data[1]
    question_hash = data[2]
    user_id = update.effective_user.id
    
    is_positive = feedback_type == "yes"
    feedback_text = f"Отзыв на вопрос с хэшем {question_hash}"
    user_data_manager.add_feedback(user_id, feedback_text, is_positive)
    
    try:
        message_text = query.message.text.split("\n\n🔹 Этот ответ был полезен?")[0]
        await query.edit_message_text(
            text=message_text,
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Error editing feedback message: {e}")
    
    emoji = "👍" if is_positive else "👎"
    await context.bot.send_message(
        chat_id=user_id,
        text=f"{emoji} Спасибо за ваш отзыв! Это поможет нам стать лучше.",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🏠 На главную", callback_data="start")]
        ]),
        parse_mode="HTML"
    )

async def cancel_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена вопроса"""
    query = update.callback_query
    await query.answer()
    
    await start(update, context)
    return SELECTING_COUNTRY

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Завершение диалога"""
    user = update.effective_user
    await update.message.reply_text(
        f"👋 До свидания, {user.first_name}! Если у вас появятся вопросы, "
        "просто напишите /start чтобы начать заново.",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode="HTML"
    )
    return ConversationHandler.END

async def send_new_start_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отправляет новое стартовое сообщение (без попыток редактирования)"""
    user = update.effective_user
    
    welcome_text = (
        f"✨ <b>Добро пожаловать, {user.first_name}!</b> ✨\n\n"
        "Я - ваш персональный гид по адаптации в новой стране. "
        "Я помогу вам с самой актуальной информацией о:\n\n"
        "• 🛂 Визовых требованиях\n"
        "• 🗣️ Языковых особенностях\n"
        "• 🆘 Экстренных ситуациях\n"
        "• 🕒 Часовых поясах\n"
        "• 💰 Валюте и финансах\n"
        "• 🎭 Культурных нормах\n\n"
        "Выберите действие ниже или просто задайте мне вопрос:"
    )
    
    keyboard = [
        [InlineKeyboardButton("🌍 Выбрать страну", callback_data="select_country")],
        [InlineKeyboardButton("❓ Задать вопрос", callback_data="ask_question")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats_all")],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=welcome_text,
        reply_markup=reply_markup,
        parse_mode="HTML"
    )
    
    logger.info(f"User {user.id} started the bot")
    return SELECTING_COUNTRY

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Улучшенный обработчик кнопок с надежной навигацией"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    logger.info(f"User {user_id} pressed button: {query.data}")
    
    try:
        if query.data == "start":
            return await send_new_start_message(update, context)
        elif query.data == "select_country":
            return await show_country_selection(update, context)
        elif query.data == "help":
            await help_command(update, context)
            return SELECTING_COUNTRY
        elif query.data.startswith("country_"):
            return await select_country(update, context)
        elif query.data.startswith("feature_"):
            if "_time_" in query.data:
                country = user_data_manager.get_country(user_id)
                if country:
                    await show_time(update, country, context)
            elif "_stats_" in query.data:
                await show_feature_stats(update, context)
            else:
                await show_feature_info(update, context)
            return SELECTING_FEATURE
        elif query.data.startswith("back_to_"):
            country = query.data.split("_")[2]
            # Вместо редактирования отправляем новое сообщение
            await show_country_features(update, country, context)
            return SELECTING_FEATURE
        elif query.data == "ask_question":
            return await ask_question_handler(update, context)
        elif query.data.startswith("feedback_"):
            await handle_feedback(update, context)
            return SELECTING_FEATURE
        elif query.data.startswith("stats_"):
            await show_country_stats(update, context)
            return SELECTING_COUNTRY
        elif query.data == "cancel_question":
            return await cancel_question(update, context)
            
    except Exception as e:
        logger.error(f"Error in button handler: {e}")
        
        # Всегда отправляем новое сообщение об ошибке
        error_text = "⚠️ Произошла ошибка. Пожалуйста, попробуйте еще раз."
        
        await context.bot.send_message(
    chat_id=user_id,
    text=error_text,
    reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 На главную", callback_data="start")]
    ]),  # Добавлена закрывающая скобка для InlineKeyboardMarkup
    parse_mode="HTML"
)
        
        return SELECTING_COUNTRY

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик ошибок"""
    error = context.error
    
    if isinstance(error, BadRequest) and "Message is not modified" in str(error):
        if update.callback_query:
            await update.callback_query.answer("Контент не изменился")
        return
    
    logger.error("Exception while handling an update:", exc_info=error)
    
    if ADMIN_IDS:
        error_trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        error_message = (
            f"⚠️ <b>Ошибка в боте:</b>\n\n"
            f"<code>{html.escape(str(error))}</code>\n\n"
            f"Update: {html.escape(str(update)) if update else 'None'}"
        )
        
        for admin_id in ADMIN_IDS:
            try:
                await context.bot.send_message(
                    chat_id=admin_id,
                    text=error_message,
                    parse_mode="HTML"
                )
            except Exception as e:
                logger.error(f"Failed to send error notification to admin {admin_id}: {e}")
    
    error_message = (
        "⚠️ Произошла непредвиденная ошибка. "
        "Разработчики уже уведомлены и работают над исправлением.\n\n"
        "Пожалуйста, попробуйте выполнить действие еще раз или вернитесь позже."
    )
    
    if update.callback_query:
        await update.callback_query.answer("Ошибка! Пожалуйста, попробуйте снова.")
        try:
            await update.callback_query.edit_message_text(
                error_message,
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🏠 На главную", callback_data="start")]
                ]),
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Error editing error message: {e}")
    elif update.message:
        await update.message.reply_text(
            error_message,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 На главную", callback_data="start")]
            ]),
            parse_mode="HTML"
        )

def main() -> None:
    """Запуск бота"""
    persistence = PicklePersistence(filepath=DATA_FILE)
    
    application = Application.builder() \
        .token(TELEGRAM_TOKEN) \
        .persistence(persistence) \
        .build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SELECTING_COUNTRY: [
                CallbackQueryHandler(select_country, pattern="^country_"),
                CallbackQueryHandler(button_handler),
                CommandHandler("help", help_command),
                CommandHandler("ask", ask_question_handler),
                CommandHandler("stats", show_country_stats),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_question)
            ],
            SELECTING_FEATURE: [
                CallbackQueryHandler(show_feature_info, pattern="^feature_"),
                CallbackQueryHandler(button_handler),
                CommandHandler("help", help_command),
                CommandHandler("ask", ask_question_handler),
                CommandHandler("stats", show_country_stats),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_question)
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        name="main_conversation",
        persistent=True
    )
    
    application.add_handler(conv_handler)
    application.add_error_handler(error_handler)
    
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stats", show_country_stats))
    
    logger.info("Бот запущен...")
    
    def stop_bot():
        logger.info("Бот останавливается...")
        application.updater.stop()
        application.stop()
    
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            close_loop=False
        )
    except KeyboardInterrupt:
        stop_bot()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        stop_bot()

if __name__ == "__main__":
    main()