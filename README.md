# OINORI

uv 使うと便利です。[リンク](https://docs.astral.sh/uv/)

## 使い方

- `cp .env.example .env`でコピーして.env に鍵とかを入れる。
- `uv run manage.py migrate`を最初のみする。
- `uv run daphne -p 8000 config.asgi:application`で起動。

http://localhost:8000
