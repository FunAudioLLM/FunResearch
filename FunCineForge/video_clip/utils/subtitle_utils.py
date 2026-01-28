#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
import re

def time_convert(ms):
    ms = int(ms)
    tail = ms % 1000
    s = ms // 1000
    mi = s // 60
    s = s % 60
    h = mi // 60
    mi = mi % 60
    h = "00" if h == 0 else str(h)
    mi = "00" if mi == 0 else str(mi)
    s = "00" if s == 0 else str(s)
    tail = str(tail).zfill(3)
    if len(h) == 1: h = '0' + h
    if len(mi) == 1: mi = '0' + mi
    if len(s) == 1: s = '0' + s
    return "{}:{}:{},{}".format(h, mi, s, tail)

def str2list(text):
    pattern = re.compile(r'[\u4e00-\u9fff]|[\w-]+', re.UNICODE)
    elements = pattern.findall(text)
    return elements

class Text2SRT():
    # 终结性标点（句尾合法结束符）
    TERMINAL_PUNCTUATION_ZH = re.compile(r'[。！？]$', re.UNICODE)
    TERMINAL_PUNCTUATION_EN = re.compile(r'[.!?]$', re.UNICODE)
    
    # 所有常见非终结性标点（需要被“升级”为句号的）
    NON_TERMINAL_PUNCTUATION = re.compile(r'[,:;，、；：]$', re.UNICODE)

    def __init__(self, text, timestamp, offset=0):
        self.token_list = text
        self.timestamp = timestamp
        start, end = timestamp[0][0] - offset, timestamp[-1][1] - offset
        self.start_sec, self.end_sec = start, end
        self.start_time = time_convert(start)
        self.end_time = time_convert(end)

    def _is_chinese_char(self, char):
        """判断字符是否为中文字符"""
        return '\u4e00' <= char <= '\u9fff'

    def _is_chinese_context(self, text):
        """判断一段文本是否属于中文语境（中文字符占优）"""
        chars = [c for c in text if c.isalnum() or self._is_chinese_char(c)]
        if not chars:
            return False
        zh_count = sum(1 for c in chars if self._is_chinese_char(c))
        return zh_count / len(chars) > 0.5

    def text(self):
        if isinstance(self.token_list, str):
            text = self.token_list.strip()
            if not text:
                return ""
            # 已以合法终结标点结尾（！？。!?）→ 保留
            if (self.TERMINAL_PUNCTUATION_ZH.search(text) or 
                self.TERMINAL_PUNCTUATION_EN.search(text)):
                return text
            # 以其他标点结尾（如 , ; : ， ； ： 、）→ 替换为句号
            if self.NON_TERMINAL_PUNCTUATION.search(text):
                is_zh = self._is_chinese_context(text)
                return re.sub(self.NON_TERMINAL_PUNCTUATION, '。' if is_zh else '.', text)
            # 无任何标点结尾 → 添加合适句号
            is_zh = self._is_chinese_context(text)
            return text + ('。' if is_zh else '.')
        else:
            res = ""
            for word in self.token_list:
                if self._is_chinese_char(word):
                    res += word
                else:
                    res += " " + word
            res = res.strip()
            if not res:
                return ""
            # 同上三种情况
            if (self.TERMINAL_PUNCTUATION_ZH.search(res) or 
                self.TERMINAL_PUNCTUATION_EN.search(res)):
                return res
            if self.NON_TERMINAL_PUNCTUATION.search(res):
                is_zh = self._is_chinese_context(res)
                return re.sub(self.NON_TERMINAL_PUNCTUATION, '。' if is_zh else '.', res)
            is_zh = self._is_chinese_context(res)
            return res + ('。' if is_zh else '.')

    def srt(self, acc_ost=0.0):
        return "{} --> {}\n{}\n".format(
            time_convert(self.start_sec + acc_ost * 1000),
            time_convert(self.end_sec + acc_ost * 1000),
            self.text()
        )

    def time(self, acc_ost=0.0):
        return (self.start_sec / 1000 + acc_ost, self.end_sec / 1000 + acc_ost)


def generate_srt(sentence_list):
    srt_total = ''
    for i, sent in enumerate(sentence_list):
        t2s = Text2SRT(sent['text'], sent['timestamp'])
        if 'spk' in sent:
            srt_total += "{}  spk{}\n{}".format(i + 1, sent['spk'], t2s.srt())
        else:
            srt_total += "{}\n{}\n".format(i + 1, t2s.srt())
    return srt_total

def generate_srt_clip(sentence_list, start, end, begin_index=0, time_acc_ost=0.0):
    start, end = int(start * 1000), int(end * 1000)
    srt_total = ''
    cc = 1 + begin_index
    subs = []
    for _, sent in enumerate(sentence_list):
        if sent['timestamp'][-1][1] <= start:
            continue
        if sent['timestamp'][0][0] >= end:
            break
        # parts in between
        if (sent['timestamp'][-1][1] <= end and sent['timestamp'][0][0] > start) or (sent['timestamp'][-1][1] == end and sent['timestamp'][0][0] == start):
            t2s = Text2SRT(sent['text'], sent['timestamp'], offset=start)
            if 'spk' in sent:
                srt_total += "{}  spk{}\n{}".format(cc, sent['spk'], t2s.srt(time_acc_ost))
            else:
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
            subs.append((t2s.time(time_acc_ost), t2s.text()))
            cc += 1
            continue
        if sent['timestamp'][0][0] <= start:
            if not sent['timestamp'][-1][1] > end:
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        break
                _text = sent['text'][j:]
                _ts = sent['timestamp'][j:]
            else:
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        _start = j
                        break
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > end:
                        _end = j
                        break
                # _text = " ".join(sent['text'][_start:_end])
                _text = sent['text'][_start:_end]
                _ts = sent['timestamp'][_start:_end]
            if len(ts):
                t2s = Text2SRT(_text, _ts, offset=start)
                if 'spk' in sent:
                    srt_total += "{}  spk{}\n{}".format(cc, sent['spk'], t2s.srt(time_acc_ost))
                else:
                    srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append((t2s.time(time_acc_ost), t2s.text()))
                cc += 1
            continue
        if sent['timestamp'][-1][1] > end:
            for j, ts in enumerate(sent['timestamp']):
                if ts[1] > end:
                    break
            _text = sent['text'][:j]
            _ts = sent['timestamp'][:j]
            if len(_ts):
                t2s = Text2SRT(_text, _ts, offset=start)
                if 'spk' in sent:
                    srt_total += "{}  spk{}\n{}".format(cc, sent['spk'], t2s.srt(time_acc_ost))
                else:
                    srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append(
                    (t2s.time(time_acc_ost), t2s.text())
                    )
                cc += 1
            continue
    return srt_total, subs, cc
