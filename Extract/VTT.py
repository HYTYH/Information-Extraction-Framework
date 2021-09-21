# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:06:07 2021

@author: worldofgoo9
"""
import argparse
import speech_recognition as sr
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default="", help='源音频文件路径')
parser.add_argument('--target', type=str, default="", help='目标存储位置路径')
args = parser.parse_args()
assert args.source!="" ,"源文件地址必须设定"

file = args.source
r = sr.Recognizer()
harvard = sr.AudioFile(file)

with harvard as source:
    # 去噪
    r.adjust_for_ambient_noise(source, duration=0.2)
    audio = r.record(source)

# 语音识别
text = r.recognize_google(audio, language="cmn-Hans-CN", show_all=False)
if(args.target==""):
    print(text)
else:
    with open(args.target,"w",encoding="utf-8") as f:
        f.write(text)




