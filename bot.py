import logging
import os
from uuid import uuid4
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, InlineQueryHandler
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# load .env variables
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "")

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def setup_rag_chain():
    logger.info("Setting up RAG chain...")

    # --- A. Load the LLM (Pointing to our OpenWebUI) ---
    # This is the key part for our setup.
    # We use ChatOpenAI, but point it to our self-hosted URL.
    llm = ChatOpenAI(
        model="gpt-5",  # Or whatever model you are using in OpenWebUI
        base_url=OPENWEBUI_URL,
        api_key=OPENWEBUI_API_KEY,
        temperature=0.7,
    )
    logger.info(f"LLM loaded. Pointing to {OPENWEBUI_URL}")

    # --- B. Load and Process FAQ Data ---
    # We'll load the "answer" as the main content.
    loader = JSONLoader(
        file_path='./faqs.json',
        jq_schema='.[]',
        content_key="answer",
        metadata_func=lambda record, metadata: {
            "source": "faq",
            "question": record.get("question"),
            "category": record.get("category")
        }
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} FAQ documents.")

    # --- C. Create Embeddings and Vector Store ---
    # This runs 100% locally.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # This creates the in-memory vector database
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # This creates a "retriever" that finds the most relevant FAQ
    retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Get top 2 results
    logger.info("Vector store and retriever created.")

    # --- D. Create the Prompt Template ---
    # This is where you tell the LLM how to be "human"
    template = """
    You are a friendly and helpful support assistant.
    Answer the user's question in a conversational way based ONLY on the following context.
    If you don't know the answer from the context, just say "I'm not sure I have that information, but I can check for you."

    CONTEXT:
    {context}

    USER QUESTION:
    {input}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # --- E. Combine it all into a "Chain" ---
    # This is the magic of LangChain (LCEL)
    
    def format_docs(docs):
        # Combine the retrieved FAQ answers into a single block
        return "\n\n".join(f"FAQ Answer: {doc.page_content} (Related to: {doc.metadata['question']})" for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("RAG chain setup complete.")
    return rag_chain

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a welcome message when the /start command is issued."""
    await update.message.reply_text("Hi! I'm our friendly FAQ bot. Ask me anything about our service!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all non-command text messages and sends them to the RAG chain."""
    user_text = update.message.text
    rag_chain = context.bot_data["rag_chain"] # Get the chain from bot context

    logger.info(f"Received message: {user_text}")
    
    # Show a "typing..." notification
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    
    # --- HERE'S THE MAGIC ---
    # We "invoke" the chain with the user's text.
    # The chain automatically does:
    # 1. Embeds the user_text
    # 2. Searches FAISS for matching FAQs
    # 3. Formats the prompt
    # 4. Calls our OpenWebUI LLM
    # 5. Gets the final "human" answer
    response = rag_chain.invoke(user_text)

    await update.message.reply_text(response)

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
    rag_chain = setup_rag_chain()

    start_handler = CommandHandler('start', start)
    rag_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    caps_handler = CommandHandler('caps', caps)
    inline_caps_handler = InlineQueryHandler(inline_caps)
    document_handler = MessageHandler(filters.PHOTO | filters.Document.PDF | filters.VIDEO, document)
    unknown_handler = MessageHandler(filters.COMMAND, unknown)

    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.bot_data["rag_chain"] = rag_chain

    application.add_handler(start_handler)
    application.add_handler(rag_handler)
    application.add_handler(caps_handler)
    application.add_handler(inline_caps_handler)
    application.add_handler(document_handler)
    application.add_handler(unknown_handler)

    # Run the bot
    logger.info("Starting bot polling...")
    application.run_polling()
if __name__ == '__main__':
    main()
    