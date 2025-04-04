import random
import socket


class IPUtil:
    @staticmethod
    def check_port_in_use(host: str, port: int) -> bool:
        """检查指定主机的端口是否被占用

        Args:
            host: 目标主机的IP地址或域名
            port: 要检查的端口号

        Returns:
            bool: 如果端口被占用返回True，否则返回False
        """

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # 设置连接超时时间为1秒
                s.settimeout(1)
                # 尝试连接指定主机和端口
                result = s.connect_ex((host, port))
                # 如果返回0表示端口被占用
                return result == 0
        except Exception as e:
            print(f"检查端口时发生错误: {e}")
            return False

    @staticmethod
    def get_available_port(host: str, start_port: int = 30000, end_port: int = 40000) -> int:
        """获取一个可用的端口号"""
        random_port = random.randint(start_port, end_port)
        while IPUtil.check_port_in_use(host, random_port):
            random_port = random.randint(start_port, end_port)
        return random_port
