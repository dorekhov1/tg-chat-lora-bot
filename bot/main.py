import logging
import os

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class TelegramBot:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.http_client = httpx.AsyncClient()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Hello! Use /generate <prompt> to generate text."
        )

    async def generate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text(
                "Please provide a prompt. Example: /generate Hello, how are you?"
            )
            return

        prompt = " ".join(context.args)
        try:
            response = await self.http_client.post(
                f"{self.api_url}/generate", json={"prompt": prompt}
            )
            response.raise_for_status()
            generated_text = response.json()["text"]
            await update.message.reply_text(generated_text)
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)}")

    async def cleanup(self):
        await self.http_client.aclose()


def main():
    api_url = os.getenv("API_URL", "http://api:8027")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

    bot = TelegramBot(api_url)
    application = Application.builder().token(bot_token).build()

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("generate", bot.generate))

    application.run_polling()


if __name__ == "__main__":
    main()
