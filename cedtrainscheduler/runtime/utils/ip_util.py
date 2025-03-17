import logging
import socket


def get_ip_address() -> str:
    """获取本机的非本地回环 IP 地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到Google的DNS服务器，不会真正建立连接
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logging.error(f"Error getting IP address: {e}")
        return "127.0.0.1"
