from fastapi import FastAPI
from langserve import add_routes

from simple_message_with_template import SimpleMessageWithTemplate

app = FastAPI(
    title="Translator App!",
    version="1.0.0",
    description="This is a chat bot"
)
bot = SimpleMessageWithTemplate()
add_routes(app, bot.chain, path="/chain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3500)
