import re


def emoji_filter(text):
    return re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+',
        '', text)


def japanese_stream_filter(text):
    for filter_pattern in [
            r'【.+】', r'ご視聴ありがとうございました', r'チャンネル登録をお願いいたします', r'ご視聴いただきありがとうございます', r'チャンネル登録してね',
            r'字幕視聴ありがとうございました', r'動画をご覧頂きましてありがとうございました', r'次の動画でお会いしましょう', r'最後までご視聴頂きありがとうございました',
            r'次の動画もお楽しみに', r'次回もお楽しみに', r'また次回の動画でお会いしましょう', r'ご覧いただきありがとうございます'
    ]:
        text = re.sub(filter_pattern, '', text)

    for filter_text in ['エンディング', '字幕作成', 'この動画の字幕']:
        if filter_text in text:
            print("filter", text)
            return ""

    if len(text) < 3:
        return ''
    return text
