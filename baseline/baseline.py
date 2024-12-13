import subprocess
import os

# 定义要执行的脚本及其路径和参数
scripts = [
    {
        "path": ".", 
        "script": "image_enhancement.py", 
        "args": ["--gamma", "0.6"]  # 传递 gamma 参数
    },
    {
        "path": "zero_dce", 
        "script": "lowlight_test.py", 
        "args": []  # 无参数
    }
]

for script_info in scripts:
    script_path = script_info["path"]  # 脚本所在目录
    script = script_info["script"]  # 脚本文件名
    args = script_info["args"]  # 脚本参数

    print(f"Running {script} in {script_path} with args: {args}...")
    
    # 设置工作目录并运行脚本
    result = subprocess.run(
        ["python", script, *args],  # 将脚本和参数组合
        cwd=script_path,  # 指定工作目录
        capture_output=True,
        text=True
    )
    
    print(f"Output from {script}:\n{result.stdout}")
    if result.stderr:
        print(f"Errors from {script}:\n{result.stderr}")
    print("-" * 40)
