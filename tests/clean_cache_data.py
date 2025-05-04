import os
import time
from datetime import datetime, timedelta

# 缓存目录
cache_dir = r'D:\temp\catch_temp\joblib\sklearn\pipeline\_fit_transform_one'

# 获取当前时间
now = time.time()

# 设置保留时间（1天）
max_age = 4 * 60 * 60  # 1天以秒为单位

# 遍历缓存目录
for root, dirs, files in os.walk(cache_dir):
    for file in files:
        file_path = os.path.join(root, file)
        # 获取文件的修改时间
        file_mtime = os.path.getmtime(file_path)
        # 判断文件是否超过保留时间
        if (now - file_mtime) > max_age:
            os.remove(file_path)
            print(f"Deleted {file_path}")

print("Cache cleanup completed.")
