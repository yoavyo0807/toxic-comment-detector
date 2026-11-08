import os
from telegram.ext import Updater, MessageHandler, CallbackContext, Filters
from telegram import Update
from model_wrapper_loader import MultiLabelDetectionModel, load_model_wrapper

model_wrapper = load_model_wrapper()
token_bot = os.environ["TELEGRAM_TOKEN"]  # set in your environment, never in code



def get_full_name(message):
    first_name = message.from_user.first_name
    first_name = first_name[0].upper() + first_name[1:]
    last_name = message.from_user.last_name
    last_name = last_name[0].upper() + last_name[1:]

    return first_name + ' ' + last_name


# Define a function to handle the messages that the bot receives
def text_handler(update: Update, context: CallbackContext):
    # Get the message from the update
    message = update.message

    str_prediction = model_wrapper.predict(message.text)
    if str_prediction != "This message was approved":
        full_name = get_full_name(message)
        str_prediction = full_name + ': ' + str_prediction.lower()
        message.delete()
        context.bot.send_message(chat_id=update.message.chat_id,
                                 text=str_prediction)
        return


def photo_handler(update: Update, context: CallbackContext):
    message = update.message
    full_name = get_full_name(message)
    output_str = f"{full_name}: Pictures are not allowed"
    message.delete()
    context.bot.send_message(chat_id=update.message.chat_id,
                             text=output_str)


def video_handler(update: Update, context: CallbackContext):
    message = update.message
    full_name = get_full_name(message)
    output_str = f"{full_name}: Videos are not allowed"
    message.delete()
    context.bot.send_message(chat_id=update.message.chat_id,
                             text=output_str)


def audio_handler(update: Update, context: CallbackContext):
    message = update.message
    full_name = get_full_name(message)
    output_str = f"{full_name}: Audio is not allowed"
    message.delete()
    context.bot.send_message(chat_id=update.message.chat_id,
                             text=output_str)


def voice_handler(update: Update, context: CallbackContext):
    message = update.message
    full_name = get_full_name(message)
    output_str = f"{full_name}: Voice is not allowed"
    message.delete()
    context.bot.send_message(chat_id=update.message.chat_id,
                             text=output_str)


def main() -> None:
    updater = Updater(token_bot, use_context=True)
    updater.dispatcher.add_handler(MessageHandler(Filters.text, text_handler))
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, photo_handler))
    updater.dispatcher.add_handler(MessageHandler(Filters.video, video_handler))
    updater.dispatcher.add_handler(MessageHandler(Filters.audio, audio_handler))
    updater.dispatcher.add_handler(MessageHandler(Filters.voice, voice_handler))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
