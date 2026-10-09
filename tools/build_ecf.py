"""编译仓库已有 ECF 解码器；算法无修改，只将临时依赖变成可复现依赖。"""
import argparse
import platform
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',default='c++',help='可用的 C++17 编译器命令')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/'third_party/ecf';destination=root/'lib';destination.mkdir(exist_ok=True)
    suffix='.dylib' if platform.system()=='Darwin' else '.so'
    output=destination/('libecf_decode'+suffix)
    command=[args.compiler,'-std=c++17','-O3','-fPIC','-shared','-I',str(source),
             str(source/'decode.cpp'),str(source/'ecf_codec.cpp'),'-o',str(output)]
    subprocess.run(command,check=True)
    print(output)


if __name__=='__main__':main()
