import logging
import os
from uuid import uuid4
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, InlineQueryHandler
from dotenv import load_dotenv

# load .env variables
load_dotenv()
bot_token = os.getenv("BOT_TOKEN", "")

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

async def caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_caps = ' '.join(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)
    
async def inline_caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query
    if not query:
        return
    results = []
    results.append(
        InlineQueryResultArticle(
            id=str(uuid4()),
            title='Caps',
            input_message_content=InputTextMessageContent(query.upper())
        )
    )
    await context.bot.answer_inline_query(update.inline_query.id, results)

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")

async def document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.document:
        file = await update.message.document.get_file()
        file_name = update.message.document.file_name
        await file.download_to_drive(file_name)
    elif update.message.photo:
        # Get the largest photo size
        file = await update.message.photo[-1].get_file()
        file_name = f"photo_{file.file_unique_id}.jpg" # Create a unique name for photos
        await file.download_to_drive(file_name)
    elif update.message.video:
        file = await update.message.video.get_file()
        file_name = update.message.video.file_name
        await file.download_to_drive(file_name)
    else:
        await update.message.reply_text("Please send a document, photo, or video.")
        return
    
def main() -> None:
    start_handler = CommandHandler('start', start)
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    caps_handler = CommandHandler('caps', caps)
    inline_caps_handler = InlineQueryHandler(inline_caps)
    document_handler = MessageHandler(filters.PHOTO | filters.Document.PDF | filters.VIDEO, document)
    unknown_handler = MessageHandler(filters.COMMAND, unknown)

    application = ApplicationBuilder().token(bot_token).build()

    application.add_handler(start_handler)
    application.add_handler(echo_handler)
    application.add_handler(caps_handler)
    application.add_handler(inline_caps_handler)
    application.add_handler(document_handler)
    application.add_handler(unknown_handler)

    # Run the bot
    logger.info("Starting bot polling...")
    application.run_polling()
if __name__ == '__main__':
    main()
    