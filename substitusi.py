#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:36:30 2018

@author: budi
"""

import re
import html

pengganti = {
        '_':' ',
        '\n':' ',
       '(Rp)( |)\d+(.| )\d+\\b':'nilaiuang',
        '\\b\d+( |)=( |)\d+(rb|ribu|.\d+)\\b':'hargabarang',
        '(\d+)((\+)|)(\d+)(\')':'',
        '(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)':'urladdr',
        '(\W|^)urladdr$':'',
        '\\brt\\b':'',
        '\\b(0|[1-9]\d*)-(0|[1-9]\d*)-(0|[1-9]\d*)\\b':'formasibola',
        '\\b(\d+)(\s|)(-|(vs))(\s|)(\d+)\\b':'skor',
        '@.*?(?=\s|$)':'',
        '\\b[\w\-.]+?@\w+?\.\w{2,4}\\b':'emailaddr',   
        '[^a-zA-Z0-9 :]':'',
        '\\b0\d+( |.|)\d+(|.| )\d+(|.| )\d+\\b':' telp',   
        '(\\b\d+\,\d+\\b)|(\\b\d+\\b)':'nilaiangka',
        '\\bg+o+l+\\b':'gol',
        '\\b(n|ng|ka|)ga+(k+|)\\b':'tidak',
        '\\b(temen)\w+\\b':'teman',
        '\\b(keren)\w+\\b':'keren',
        '\\b(cepet)\\b':'cepat',
        '\\b(kudu)\\b':'harus',
        '\\b(yg)\\b':'yang',
        '\\b(kq)|(ko)\\b':'kok',
        '\\b(menang+)\\b':'menang',
        '\\b(gw)|(gue)\\b':'saya',
        '\\b(s|)(u|)d(a|)(h|)\\b':'sudah',
        '\\b(brenti)\\b':'berhenti',
        '\\bg+(e+|a+)s+\\b':'teman',
        '\\b(ha|a|)y+(u+|o+)(k+|)(s+|)\\b':'ayo',
        "\\b(nilaiangka)\\b":'',
        '\\b(di)\\b':'',
        '\\b(dan)\\b':'',
        '\\b(ini)\\b':'',
        '[^\w\d\s]':'',
        '\s+':' ',
        '^\s+|\s+?$':' '
        }
hapus = {
        '\\b(ganjartaktakutpakdirman)\\b':'',
        '\\b(persijaday)\\b':'',
        '\\b(harikebangkitannasional)\\b':'',
        '\\b(nasionalismezamannow)\\b':'',
        '\\b(tarawihinstagramable)\\b':''
        }

def sub_clean(text,pengganti):
    text = html.unescape(text)
    text = text.lower()
    text = text.strip()
    for k,v in pengganti.items():
        text=re.sub(k,v,text)
    return text

def wordfilter(text):
    text = text.lower()
    text = text.strip()
    for k,v in hapus.items():
        text=re.sub(k,v,text)
    return text

