#!/bin/bash
# Chạy trong Debian (proot-distro). Cần cài trước: apt install -y ffmpeg
#
# Cách dùng: bash start_live.sh
# Dừng bằng Ctrl+C (dừng cả ffmpeg lẫn server local)

set -e

# Đọc RTSP_URL từ config.py (đỡ phải gõ 2 nơi)
RTSP_URL=$(python3 -c "from config import RTSP_URL; print(RTSP_URL)")

mkdir -p hls
rm -f hls/*.ts hls/*.m3u8

echo "Bắt đầu chuyển RTSP -> HLS..."
ffmpeg -rtsp_transport tcp -i "$RTSP_URL" \
    -c copy -f hls \
    -hls_time 2 -hls_list_size 6 \
    -hls_flags delete_segments+append_list \
    hls/stream.m3u8 &
FFMPEG_PID=$!

echo "Bắt đầu server local ở cổng 8080..."
cd hls && python3 -m http.server 8080 &
SERVER_PID=$!

trap "kill $FFMPEG_PID $SERVER_PID 2>/dev/null" EXIT INT TERM

echo "Đã sẵn sàng. Chạy: ngrok http --domain=<ten_static_domain_cua_ban> 8080 ở 1 cửa sổ Termux khác."
wait
