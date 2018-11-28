#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:36:30 2018

@author: budi
"""
import re
import html

pengganti = {
        '(Rp)( |)\d+(.| )\d+\\b':'nilaiuang',
        '\\b\d+( |)=( |)\d+(rb|ribu|.\d+)\\b':'hargabarang',
        '(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)':'urladdr',
        '(\W|^)urladdr$':' lasturladdr',
        '\\b(0|[1-9]\d*)-(0|[1-9]\d*)-(0|[1-9]\d*)\\b':'formasibola',
        '\\b(\d+)(\s|)(-|(vs))(\s|)(\d+)\\b':'skor',
        '\\b[\w\-.]+?@\w+?\.\w{2,4}\\b':'emailaddr',   
        '\\b0\d+( |.|)\d+(|.| )\d+(|.| )\d+\\b':' telp',   
        '(\\b\d+\,\d+\\b)|(\\b\d+\\b)':' ',
        '\\bg+o+l+\\b':'gol',
        '\\b(n|ng|ka|)ga+(k+|)\\b':'tidak',
        '\\b(temen)\w+\\b':'teman',
        '\\b(keren)\w+\\b':'keren',
        '\\b(cepet)\\b':'cepat',
        '\\b(kudu)\\b':'harus',
        '\\b(lg)\\b':'lagi',
        '\\b(yg)\\b':'yang',
        '\\b(nanya)\\b':'bertanya',
        '\\b(kq)|(ko)\\b':'kok',
        '\\b(menang+)\\b':'menang',
        '\\b(gw)|(gue)\\b':'saya',
        '\\b(s|)(u|)d(a|)(h|)\\b':'sudah',
        '\\b(brenti)\\b':'berhenti',
        '\\bg+(e+|a+)s+\\b':'teman',
        '\\b(ha|a|)y+(u+|o+)(k+|)(s+|)\\b':'ayo'
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

