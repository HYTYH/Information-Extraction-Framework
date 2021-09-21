# -*- coding: utf-8 -*-
"""
Created on Thu May 27 23:08:25 2021

@author: worldofgoo9
"""

import argparse
import sys
import os
#import TextInfoExtract as te

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default="", help='源音频文件路径')
parser.add_argument('--target', type=str, default="", help='目标存储位置路径')
args = parser.parse_args()

assert args.source!="" ,"源文件地址必须设定"



    
def main():
    print("Current Path in VoiceExtract ",os.getcwd())

    with open("数据/TempText.txt","w",encoding="utf-8") as f:
        f.write("")

    os.system(f"python VTT.py --source=\"{args.source}\" --target=\"数据/TempText.txt\"")
    
    os.system(f"python TextInfoExtract.py --all --specificdoc=\"TempText\"")
    
    with open('提取结果/信息/结果指标.txt','r',encoding="utf-8") as f:
        res0=f.read()
    with open('提取结果/信息/TempText.txt','r',encoding='utf-8') as f:
        res1=f.read()
    
    with open(args.target,'w',encoding="utf-8") as f:
        f.write(res1+res0)
    targetJPG=args.target+".jpg"
    #os.system(f"copy 提取结果\关系网络\TempText.jpg {targetJPG}")
    with open('提取结果/关系网络/TempText.jpg','rb') as f:
        imageBytes=f.read()
    with open(targetJPG,'wb') as f:
        f.write(imageBytes)
    

    
if __name__ == '__main__':
    
    main()
