import os

# 缓存目录
cache_dir = '../my_cache_directory'

# 设置保留的最大缓存大小（100GB）
max_cache_size = 100 * 1024 * 1024 * 1024  # 100GB in bytes

# 获取所有缓存文件的信息
file_info = []
for root, dirs, files in os.walk(cache_dir):
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path)
        file_mtime = os.path.getmtime(file_path)
        file_info.append((file_path, file_size, file_mtime))

# 按修改时间排序（最近的在前）
file_info.sort(key=lambda x: x[2], reverse=True)

# 保留最新的文件，总大小不超过max_cache_size
current_size = 0
files_to_keep = []

for file_path, file_size, file_mtime in file_info:
    if current_size + file_size <= max_cache_size:
        current_size += file_size
        files_to_keep.append(file_path)
    else:
        break

# 删除不需要保留的文件
files_to_delete = set(file_info) - set(files_to_keep)

for file_path, _, _ in files_to_delete:
    os.remove(file_path)
    print(f"Deleted {file_path}")

print("Cache cleanup completed.")
