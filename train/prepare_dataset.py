import json
from datetime import datetime, timedelta
from typing import List, Literal, Optional

from loguru import logger
from pydantic import BaseModel


class Message(BaseModel):
    date: datetime
    author: str
    text: str


class Chat(BaseModel):
    name: str
    type: Literal["personal_chat", "private_group", "private_supergroup"]
    messages: List[Message]
    sessions: Optional[List[List[Message]]] = []


def load_chats(path: str) -> list[Chat]:
    chats: List[Chat] = []
    logger.info(f"Loading chats from '{path}'...")
    with open(path, "r") as f:
        chat = json.load(f)
        messages = [
            Message(
                date=msg["date"],
                author=msg["from"],
                text="".join(
                    [text_entity["text"] for text_entity in msg["text_entities"]]
                )
                + msg.get("sticker_emoji", ""),
            )
            for msg in chat["messages"]
            if "from" in msg
            and msg["from"]
            and (msg["text_entities"] or "sticker_emoji" in msg)
        ]
        if messages:
            chat = Chat(name=chat["name"], type=chat["type"], messages=messages)
            chats.append(chat)
    logger.info(f"Found {len(chats)} chats in file '{path}'")
    return chats


def transform_chats(
    input: str,
    output: str,
    last_x_months: int = 120,
    session_minutes_threshold: int = 10,
    concat_one_user_messages_delimeter: str = "\n>>> ",
):
    chats = load_chats(input)

    for chat in chats:
        chat.messages = [
            msg
            for msg in chat.messages
            if msg.date > datetime.now() - timedelta(days=last_x_months * 30)
        ]
    chats = [chat for chat in chats if chat.messages]
    logger.info(f"After filtering by date, there are {len(chats)} chats left")

    for chat in chats:
        sessions = []
        current_session = []
        for msg in chat.messages:
            if (
                not current_session
                or (msg.date - current_session[-1].date).seconds / 60
                < session_minutes_threshold
            ):
                current_session.append(msg)
            else:
                sessions.append(current_session)
                current_session = [msg]
        if current_session:
            sessions.append(current_session)
        chat.sessions = sessions
    logger.info("Combined messages into sessions")

    for chat in chats:
        sessions = []
        for session in chat.sessions:
            current_session = []
            current_message = session[0]
            current_message.text = (
                concat_one_user_messages_delimeter.lstrip() + current_message.text
            )
            for msg in session[1:]:
                if msg.author == current_message.author:
                    current_message.text += (
                        concat_one_user_messages_delimeter + msg.text
                    )
                else:
                    current_session.append(current_message)
                    current_message = msg
                    current_message.text = (
                        concat_one_user_messages_delimeter.lstrip()
                        + current_message.text
                    )
            current_session.append(current_message)
            sessions.append(current_session)
        chat.sessions = sessions
    logger.info("Combined consecutive messages from single user into one message")

    for chat in chats:
        for session in chat.sessions:
            for msg in session:
                del msg.date

    all_sessions = []
    for chat in chats:
        for session in chat.sessions:
            all_sessions.append(session)

    with open(output, "w") as f:
        json.dump(
            [
                [{"author": msg.author, "text": msg.text} for msg in session]
                for session in all_sessions
            ],
            f,
            indent=4,
            ensure_ascii=False,
        )
    logger.info(
        f"Took {len(all_sessions)} sessions from {len(chats)} chats and wrote them to '{output}'. Average session length is {round(sum(len(session) for session in all_sessions) / len(all_sessions), 2)} messages"
    )


if __name__ == "__main__":
    transform_chats(".data/result.json", ".data/result_sessions.json")
