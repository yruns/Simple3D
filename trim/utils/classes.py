import sys
import os

class Tee:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, message):
        for output in self.outputs:
            output.write(message)

    def flush(self):
        for output in self.outputs:
            if not output.closed:
                output.flush()

if __name__ == '__main__':

    # 打开文件以写入
    log_file = open('output.log', 'w')

    # 创建 Tee 实例，将标准输出流和文件输出流传入
    sys.stdout = Tee(sys.stdout, log_file)

    # 现在所有的 print 输出将同时显示在控制台和写入 output.log 文件
    print("Hello, World!")
    print("This is a test message.")

    # 关闭文件
    log_file.close()