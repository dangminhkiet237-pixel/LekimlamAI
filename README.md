# Camera AI - EZVIZ C6N + Telegram

## Cài đặt (trong Termux → proot-distro Debian)

```bash
git clone <link_repo_cua_ban>.git
cd <ten_repo>
cp config.example.py config.py
nano config.py   # điền RTSP_URL, TELEGRAM_TOKEN, CHAT_ID thật vào đây

pip install -r requirements.txt
git clone https://github.com/chuanqi305/MobileNet-SSD.git models/mobilenet-ssd
mkdir -p known_faces
# thêm known_faces/<ten_nguoi>/anh1.jpg, anh2.jpg, ...

python3 detect_advanced.py
```

## Cập nhật code sau này

```bash
git pull
```

`config.py`, `models/`, `known_faces/` không nằm trong repo (xem `.gitignore`) —
mỗi máy tự tạo lại các phần này, không bị ghi đè khi `git pull`.
